import bpy
from random import seed, randint, uniform  # , getrandbits
from math import sin, cos, copysign
# from numpy import log as ln
from os.path import join
from time import sleep
from datetime import datetime
import argparse
import sys
#from mathutils import Vector #***********
emitter_locations = []
emitter_coordinates = []
#*************


seed()

scenes = 8  # Measures how many scenes are randomly set and rendered
fp_scene = 4  # Frames-per-scene, how many pictures are rendered per scene

domain = bpy.context.scene.objects["Smoke Domain"]
emitter = bpy.context.scene.objects["Emitter"]
init_velocity = emitter.modifiers["Fluid"].flow_settings.velocity_coord
light = bpy.context.scene.objects["Light"]
cam = bpy.context.scene.objects["Camera"]
turbulence = bpy.context.scene.objects["Turbulence"]
wind = bpy.context.scene.objects["Wind"]

output_dir = ''  # = '/Users/Max/Documents/Blender/Smokes/trials/render{}'
folder_num = ''

sign = lambda x: copysign(1, x)
start_time = datetime.now()


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is taken online from stack exhange. Use double dashes to express arguments
    given after the script in the console
    >> blender --python my_script.py -- -a 1 -b 2
    """

    def _get_argv_after_doubledash(self):
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self, args=None, namespace=None):
        return super().parse_args(args=self._get_argv_after_doubledash())


def rand_color_val() -> float:
    """
    Returns a number between 0 and 0.6 generated with weight, to represent RGB colors

    :return:
    """

    # Generate num between 0 and 1
    c = randint(0, 100) / 100

    # Put it through a weighted function
    return max(c/5, 54*c/70 - 6/35)

    # return min(0.6, max(0, -0.3 + c))
    # Put it through a weighted piecewise function
    # if c >= 0.83:
    #     return 0.6
    # else:
    #     c = -(ln(-0.6*c + 0.5) + 0.64) / 10
    #     return round(c, 2)


def rand_smoke_offset() -> float:
    """
    Makes an x-offset, away from the camera, for the emitter's transform to emulate
    smaller smoke 
    
    Pushes the smoke back further into the frame
    """
    
    if randint(0, 1) == 0: 
        return 0
    
    else:
        return randint(-850, -300) / 100  # -8.5 to -3, pushes the smoke back


def setup_new_scene():
    """
    Sets up a new random configuration for a scene

    :return:
    """

    # Domain Shader (Color and Density)
    col = rand_color_val()  # RGB between 0 and .6
    domain.active_material.node_tree.nodes['Principled Volume'].inputs[0].default_value = (col, col, col, 1)

    min_den = 1 / (2 * col + 0.5)
    max_den = -0.7 * col + 0.5
    density = uniform(min_den, max_den)  # Random density within a range depending on col
    domain.active_material.node_tree.nodes['Principled Volume'].inputs[2].default_value = density

    # Domain Settings
    domain.modifiers['Fluid'].domain_settings.vorticity = randint(30, 60) / 100  # 0.3 to 0.6

    # Emitter Initial Velocity
    init_vel_degree = randint(0, 6283) / 1000
    init_vel_degree2 = randint(0, 3141) / 1000  # A higher value means it will be directed downward
    init_vel_mag = randint(2, 60) / 10  # 0.2 to 6

    init_velocity.x = init_vel_mag * sin(init_vel_degree2) * cos(init_vel_degree)
    init_velocity.y = init_vel_mag * sin(init_vel_degree2) * sin(init_vel_degree)
    init_velocity.z = init_vel_mag * cos(init_vel_degree2)

    # Turbulence Transform -- Needs to be near the emitter
    turbulence.location.x = domain.location.x + randint(-32, 32) / 10
    turbulence.location.y = domain.location.y + randint(-32, 32) / 10
    turbulence.location.z = domain.location.z + randint(-16, 16) / 10

    # Turbulence Field
    turbulence.field.strength = randint(300, 700) / 100  # 3 to 7
    turbulence.field.flow = randint(0, 100) / 100  # 0 to 1
    turbulence.field.size = randint(80, 600) / 100  # .8 to 6

    # Wind Field
    wind.field.flow = randint(80, 280) / 100  # 0.8 to 2.8
    wind.field.noise = randint(0, 250) / 100  # 0 to 2.5    

    # Light Transform
    light_degree = randint(524, 2618) / 1000  # Goes from 30 degrees to 150
    light.location.x = 8 * cos(light_degree)
    light.location.y = 8 * sin(light_degree)
    light.location.z = randint(500, 700) / 100

    # Light Strength
    light.data.energy = randint(800, 1200) / 100  # 8 to 12

    # ------------ Position Affecting Parameters ------------ #

    # Buoyancy and Heat -- affects Z
    domain.modifiers['Fluid'].domain_settings.alpha = randint(8, 35) / 10  # 0.8 to 3.5
    domain.modifiers['Fluid'].domain_settings.beta = randint(-35, 35) / 10  # -3.5 to 3.5
    emitter.modifiers["Fluid"].flow_settings.temperature = randint(-80, 80) / 10  # -8 to 8

    heat_sign = sign(domain.modifiers['Fluid'].domain_settings.beta) * \
                sign(emitter.modifiers["Fluid"].flow_settings.temperature)

    # Generate random numbers
    emitter_distance = randint(0, 1000) / 100  # 0 to 10
    rand_place_degree = randint(0, 6283) / 1000
    rand_adjustment = randint(-1047, 1047) / 1000  # Spans 60 degrees to the left or the right

    # Emitter Transform
    emitter.location.x = emitter_distance * cos(rand_place_degree)
    emitter.location.y = emitter_distance * sin(rand_place_degree) + rand_smoke_offset()

    # Print emitter location **************
    # print("Emitter Location:", emitter.location)

    if heat_sign == 1:
        # Smoke will likely rise
        emitter.location.z = randint(-320, 500) / 100  # -3.2 to 5
    else:
        # Smoke will likely fall
        emitter.location.z = randint(250, 1220) / 100  # 2.5 to 12.2

    # Domain Transform
    domain.location.x = emitter.location.x
    domain.location.y = emitter.location.y
    domain.location.z = emitter.location.z  # + 4.6  # The Z should be tied to emitter

    # Wind Rotation
    if emitter_distance < 4:  
        # If the emitter is within frame, then the wind can blow in any dir
        wind.rotation_euler.z = randint(0, 6283) / 1000
    
    else:
        # Otherwise, wind must blow into frame
        wind.rotation_euler.z = rand_place_degree + 3.14 + rand_adjustment
    
    wind.rotation_euler.x = rand_adjustment / 2
    wind.rotation_euler.y = 90 + -rand_adjustment / 2

    if emitter_distance > 4.2:
        # The emitter is so far that the wind should certainly blow strong enough
        wind.field.strength = randint(400, 600) / 100  # 4 to 6
    else:
        wind.field.strength = randint(0, 180) / 100  # 0 to 1.8


def delete_bake():
    bpy.context.view_layer.objects.active = domain
    bpy_freeing = bpy.ops.fluid.free_all()

    assert ('FINISHED' in bpy_freeing)


def bake():
    bpy.context.view_layer.objects.active = domain
    bpy_baking = bpy.ops.fluid.bake_all()

    assert ('FINISHED' in bpy_baking)

    # Increment and continue baking
    # domain.modifiers["Fluid"].domain_settings.cache_frame_end += 40
    # bpy_baking = bpy.ops.fluid.bake_all()
    # assert ('FINISHED' in bpy_baking)

#*****************
def emitter_location_to_image_coordinates(scene, camera, emitter_location):
    """
    Convert emitter location to image coordinates (x, y) of the rendered output.
    
    :param scene: The scene containing the camera
    :param camera: The camera object
    :param emitter_location: The world coordinates of the emitter location
    :return: The image coordinates (x, y)
    """
    # Set the active camera
    scene.camera = camera

    # Get the render resolution
    resolution_x = scene.render.resolution_x
    resolution_y = scene.render.resolution_y

    # Get the camera view matrix
    view_matrix = camera.matrix_world.normalized().inverted()

    # Project the emitter location onto the camera view
    emitter_coords = view_matrix @ emitter_location

    # Calculate the image coordinates
    image_x = int((emitter_coords.x / emitter_coords.z + 1.0) / 2.0 * resolution_x)
    image_y = int((emitter_coords.y / emitter_coords.z + 1.0) / 2.0 * resolution_y)

    return image_x, image_y

#************

def render_images(r_num, f=folder_num) -> int:
    """
    Renders four frames of the animation, subdivided into 16-20 frames per image

    :param r_num: The number of the file attached to the rendered image
    :param f: The number of the folder
    :return: Returns the new number of the file attached to the rendered image
    """

    frame = 0

    for i in range(fp_scene):

        # Set frame
        frame += randint(16, 20)
        bpy.context.scene.frame_set(frame)

        # Print emitter location **************
        print(f"Emitter Location: {emitter.location}")
        emitter_locations.append(emitter.location)

        # Print x,y location
        co = emitter_location_to_image_coordinates(bpy.context.scene, cam, emitter.location)
        print(f"Pixel coordinants: {co}")
        emitter_coordinates.append(co)





        #**************

        # Save image
        bpy.context.scene.render.filepath = join(output_dir, 'render{0:0>5}.png'.format(r_num))        
        bpy.ops.render.render(write_still=True)
        r_num += 1

    return r_num


def init_values():
    parser = ArgumentParserForBlender()

    parser.add_argument("-r", "--repeats", type=int, default=8, required=False,
                        help="How many various smoke scenes to bake.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="File location of the renders folder")
    parser.add_argument("-n", "--number", type=str, default='', required=False,
                        help="Number of the file folder")
    args = vars(parser.parse_args())

    global output_dir
    output_dir = args['output']
    global folder_num
    folder_num = args['number']
    global scenes
    scenes = args['repeats']

def init_gpu():
    """
    Ensures that blender is using Google Colab's provided GPU

    Code taken from "donmahallem" in their GitHub project colab_blender!
    """

    import re
    scene = bpy.context.scene
    scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    prefs.addons['cycles'].preferences.get_devices()
    cprefs = prefs.addons['cycles'].preferences
    print(cprefs)

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print('Device found', compute_device_type)
            break
        except TypeError:
            pass

    # Enable all CPU and GPU devices
    for device in cprefs.devices:
        if not re.match('intel', device.name, re.I):
            print('Activating', device)
            device.use = True
        else:
            device.use = False

    # bpy.ops.render.render(True)


def start_render():
    """
    Frees the smoke data, sets up new random reconfiguration, then preps to rebake / re-export

    :return:
    """

    init_values()

    assert (domain.modifiers["Fluid"].domain_settings.cache_frame_end >= fp_scene * 20)
    render_num = 1

    init_gpu()
    
    print(f'\nStarting to render {scenes * fp_scene} gas images...')

    for i in range(scenes):

        # DELETE OLD BAKE
        delete_bake()        

        # REMAKE SCENE
        setup_new_scene()

        # WAIT TO CONFIRM END OF BAKE DELETE
        sleep(0.4)

        if i != 0:
            print(f'\nCompleted render {i}, seconds: {(datetime.now() - start_time).total_seconds()}')

        # BAKE
        bake()

        # RENDER
        render_num = render_images(render_num)

    print(f'Finished producing {scenes * fp_scene} render(s), time {(datetime.now() - start_time).total_seconds()}\n')
    """
    for index, location in enumerate(emitter_locations, start=1):
      print(f"Emitter {index} Location: {location}")
    # Print emitter coordinates
    for coordinates in emitter_coordinates:
      print(f"Emitter coordinates: {coordinates}")
    """
    # Print emitter index with location and coordinates
    for index, (location, coordinates) in enumerate(zip(emitter_locations, emitter_coordinates), start=1):
      print(f"Emitter {index} Location: {location}")
      print(f"Emitter {index} Coordinates: {coordinates}")


    



if __name__ == '__main__':
    start_render()
