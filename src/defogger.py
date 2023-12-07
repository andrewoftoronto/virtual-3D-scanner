from typing import List, Tuple
from itertools import chain
import torch
from .source_loader import RawSceneData


# List of groups where each frame in the group includes a pixel that should be
# the same colour up close. They don't necessarily need to be at the same
# physical spatial location.
# Each such frame entry is (name, Y pixel, X pixel). Name can be a substring 
# of the colour map file name.
fog_inference_groups = [
    #[
    #    ('085', 1440 - 282, 402),
    #    ('058', 1440 - 523, 1348)
    #],
    [(str(i).rjust(3, '0'), 720, 1280) for i in range(0, 16)],
    [(str(i).rjust(3, '0'), 720, 1280) for i in range(16, 33)]
]

def unfog(foggy_colour, depth, fog_colour, fog_density):
    #fog_density = torch.exp(fog_density_log)
    if len(foggy_colour.shape) == 3:
        fog_colour = fog_colour.reshape([1, 1, 3])
        fog_density = fog_density.reshape([1, 1, 1])
    recovered_colour = (
        (foggy_colour - fog_colour * (1 - torch.exp(-depth * fog_density))) / 
        torch.exp(-depth * fog_density)
    )
    return recovered_colour

def fog(initial_colour, depth, fog_colour, fog_density_log):
    if len(initial_colour) == 3:
        fog_colour = fog_colour.reshape([1, 1, 3])
        fog_density_log = fog_density_log.reshape([1, 1, 1])
    fog_density = torch.exp(fog_density_log)
    interpolant = torch.exp(-depth * fog_density)
    predicted_colour = initial_colour * interpolant + fog_colour * (1 - interpolant)
    return predicted_colour, interpolant

def get_depth(depth_map_value):
    znear = 0.69
    zfar = 740
    return znear + depth_map_value * (zfar - znear)

def infer_fog_params(scene: RawSceneData):

    # Assemble frames that show the same colour.
    frame_groups = []
    for group in fog_inference_groups:
        frame_group = []
        for (image_name, y, x) in group:
            
            # TODO: This looks stupid and can use hashmaps instead.
            selected_frame = None
            for frame in chain(scene.frames, scene.extra_frames):
                if image_name in frame.colour_path:
                    selected_frame = frame
                    break
            if selected_frame is None:
                raise Exception(f"No match found for fog inference image: {image_name}")
            frame_group.append([selected_frame, y, x])

        frame_groups.append(frame_group)

    fog_colour_buf = torch.tensor([0.3608, 0.302, 0.129]).cuda()
    fog_colour_param = torch.nn.Parameter(fog_colour_buf, requires_grad=False)

    fog_density_log_buf = torch.tensor([-5.0]).cuda()
    fog_density_log_param = torch.nn.Parameter(fog_density_log_buf, requires_grad=True)

    #initial_colours_buf = torch.ones(size=[len(frame_groups), 3]).cuda() * 0.5
    initial_colours_buf = torch.tensor([[0.00784314, 0.04705882, 0.05098039], [0.03137255, 0.10588235, 0.03529412]]).cuda()
    initial_colours_param = torch.nn.Parameter(initial_colours_buf, requires_grad=False)

    # Disabled for now.
    depth_bias_buf = torch.tensor(-7.334).cuda()
    depth_bias_param = torch.nn.Parameter(depth_bias_buf, requires_grad=False)

    def objective_fn():
        optimizer.zero_grad()

        loss = 0
        total_predictions = 0
        for (i, frame_group) in enumerate(frame_groups):
            for (frame, y, x) in frame_group:

                # TODO: Codify this better.
                colour_map = torch.from_numpy(frame.get_colour()).cuda()
                colour_map = torch.flip(colour_map, dims=[0]) / 255.0

                depth_map = torch.from_numpy(frame.get_depth()).cuda()
                depth_map = torch.flip(depth_map, dims=[0])
                depth_map = get_depth(depth_map[:,:,0])

                y_start = y - 5
                y_end = y + 5
                x_start = x - 5
                x_end = x + 5
                expected_colour = colour_map[y_start:y_end,x_start:x_end,:3]
                depth = depth_map[y, x].unsqueeze(-1)
                predicted_colour, interpolant = fog(initial_colours_param[i], depth, fog_colour_param, fog_density_log_param)

                loss += torch.nn.MSELoss()(predicted_colour, expected_colour.mean(dim=[0, 1]))

                total_predictions += 1

        loss = loss / total_predictions
        loss.backward()
        print(loss.detach())

        return loss
    
    optimizer = torch.optim.Adam([fog_colour_param, fog_density_log_param, initial_colours_param, depth_bias_param], lr=1e-1)

    for t in range(0, 100):
        loss = optimizer.step(closure=objective_fn)

        # Project initial colour to be non-negative.
        initial_colours_param.data = torch.maximum(initial_colours_param.data, torch.tensor(0).cuda())

        if loss < 1.5e-5:
            break
            #depth_bias_param.requires_grad = True

        #if loss < 0.00015:
        #    break

    scene.fog_depth_bias = depth_bias_param.data
    scene.fog_colour = fog_colour_param.data
    scene.fog_density = torch.exp(fog_density_log_param.data)
    print(f"Selected fog colour and density: {scene.fog_colour} and {scene.fog_density}")
    pass

def defog(scene: RawSceneData):
    for frame in scene.frames:
        depth_map = frame.get_depth()
        colour = frame.get_colour()

        depth = get_depth(torch.from_numpy(depth_map[:,:,0]).cuda())
        depth = torch.flip(depth, dims=[0])
        depth = depth.unsqueeze(-1)

        raw_foggy = torch.from_numpy(colour[:,:,:3]).cuda()
        foggy = raw_foggy / 255.0

        unfogged = unfog(foggy, depth, scene.fog_colour, scene.fog_density).clip(0, 1)
        #unfogged = (unfogged * 1.2 + 0.1).clip(0, 1)

        unfogged = (unfogged * 255).to(torch.uint8) 
        frame.colour_map = unfogged.detach().cpu().numpy()