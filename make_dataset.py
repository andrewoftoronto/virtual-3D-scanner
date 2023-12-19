import os

def mv(old_path, new_path):
    print(f"Moving {old_path} to {new_path}")
    os.rename(old_path, new_path)

def main():
    source_folder = "C:/Program Files (x86)/Steam/steamapps/common/Valheim"
    dest_folder = "C:/Users/Andrew/colmap-projects/forest"

    # Make dest folders.
    colour_dest = os.path.join(dest_folder, "images")
    normal_dest = os.path.join(dest_folder, "normal")
    depth_dest = os.path.join(dest_folder, "depth")
    os.makedirs(colour_dest, exist_ok=True)
    os.makedirs(normal_dest, exist_ok=True)
    os.makedirs(depth_dest, exist_ok=True)

    # Identify bmp files.
    bmps = filter(lambda x: x.endswith(".bmp"), os.listdir(source_folder))

    # Check if any files already in the dataset.
    n = 0
    for file in os.listdir(colour_dest):
        n = int(file.removesuffix(".bmp")) + 1

    # Move each file to where it belongs.
    for (i, colour_file_name) in enumerate(bmps):
        base_name = colour_file_name.replace(" BackBuffer.bmp", "")
        new_base_name = str(n + i).zfill(4)

        old_colour = os.path.join(source_folder, colour_file_name)
        new_colour = os.path.join(dest_folder, "images/", f"{new_base_name}.bmp")
        mv(old_colour, new_colour)

        depth_file_name = f"{base_name} DepthBuffer.exr"
        old_depth = os.path.join(source_folder, depth_file_name)
        new_depth = os.path.join(dest_folder, "depth/", f"{new_base_name}.exr")
        mv(old_depth, new_depth)

        normal_file_name = f"{base_name} NormalMap.exr"
        old_normal = os.path.join(source_folder, normal_file_name)
        new_normal = os.path.join(dest_folder, "normal/", f"{new_base_name}.exr")
        #mv(old_normal, new_normal)


if __name__ == '__main__':
    main()

