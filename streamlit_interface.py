# @Angelica
import time
import os
from PIL import Image
from vision import Vision
# import Drone, Depth, Segmentation, Decision, Aerial, Scene_Parser
import Drone, Segmentation,Scene_Parser
import numpy as np
import streamlit as st


# start: module loaders: done this way for caching reasons
# @st.cache(hash_funcs={Aerial.Fire:id})
# def load_aerial_fire():
#     return Aerial.Fire()
#
# @st.cache(hash_funcs={Aerial.Building:id})
# def load_aerial_building():
#     return Aerial.Building()
#
# @st.cache(hash_funcs={Depth.MonoDepth2:id})
# def load_depth():
#     return Depth.MonoDepth2()

@st.cache(hash_funcs={Segmentation.UNET:id})
def load_segmentation():
    return Segmentation.UNET()

@st.cache(hash_funcs={Scene_Parser.SceneParser:id})
def load_scene_parser():
    return Scene_Parser.SceneParser()
#end: mod loaders


def select_drone():
    drone_name = st.sidebar.text_input('Running from **Tello** or **Unreal**?')
    drone_names = ['Tello', 'Unreal']
    if drone_name not in drone_names:
        st.warning('Please either Tello or Unreal')
        st.stop()

    drone_name = drone_name.lower()
    if drone_name == 'tello':
        drone = Drone.Tello()
    elif drone_name == 'unreal':
        drone = Drone.Unreal()
        drone.speed = 5

    return drone, drone_name


def select_mods():
    st.sidebar.text('_________________________________')
    st.sidebar.text('Select modules to activate')
    aerialFire_mod = st.sidebar.checkbox('Aerial Fire')
    aerialObjects_mod = st.sidebar.checkbox('Aerial Objects')
    depth_mod = st.sidebar.checkbox('Depth')
    segmentation_mod = st.sidebar.checkbox('Segmentation')

    selected_mods = []
    if aerialFire_mod:
        selected_mods.append('aerialFire_mod')
    if aerialObjects_mod:
        selected_mods.append('aerialObjects_mod')
    if depth_mod:
        selected_mods.append('depth_mod')
    if segmentation_mod:
        selected_mods.append('segmentation_mod')
    return selected_mods


def select_rl():
    st.sidebar.text('_______________________________________')
    st.sidebar.text('Select decision maker to activate')
    cv = st.sidebar.checkbox('Computer Vision')
    dq = st.sidebar.checkbox('Deep Q')
    if cv:
        return 'computer vision'
    else:
        return 'deep q'


def main():
    # RL coefficients
    coefficients = {
        'fire': 0,
        'depth': 1,
        'smoke': 0,
        'path': 1,
        'objective': 0,
        'smooth': 1
    }

    # dict to contain all user inputs
    ui = {}
    # aerial_fire = load_aerial_fire()
    # aerial_building = load_aerial_building()
    # depth = load_depth()
    segmentation = load_segmentation()
    scene_parser = load_scene_parser()

    # define modules
    visions = {
        # 'fire': aerial_fire,
        # 'depth': depth,
        'smoke':segmentation,
        'scene_parse': scene_parser
    }
    # decision = Decision.Rewards()

    # if aerial_fire and aerial_building and depth and segmentation and scene_parser:
    if segmentation and scene_parser:
        # start ui
        args = []
        st.title('Computer Vision Aided Tello Observer')

        slots = []
        used_slots = 0
        for i in range(0, 10):
            slots.append(st.empty())

        ui['user'] = st.sidebar.selectbox('Select user:', ('', 'Angelica', 'Courtney', 'Olivia', 'Tim'))
        if ui['user'] == '':
            st.warning('Insert valid user')
            st.stop()
        # select_mods(user)
        drone, drone_name = select_drone()
        ui['drone'], ui['drone_name'] = drone, drone_name



        st.subheader(f'Hi {ui["user"]}, you are using {ui["drone_name"]}')

        # select mods
        if (ui['user'] != '') and (ui['drone_name']!=''):
            ui['activated_mod'] = select_mods()
            ui['decision_mods'] = select_rl()

            # only move forward to show further element after these elements are set
            ### start main screen
            st.subheader('Flight parameters')
            col1, col2 = st.beta_columns(2)
            with col1:
                ui['distance_ts'] = st.text_input('Insert distance to cover per timestep')
                if not ui['distance_ts']: #TODO implement regex
                    st.warning('This field cannot be null')
                    st.stop()
            with col2:
                ui['drone_speed'] = st.text_input('Insert drone speed', value=5)
                if not ui['drone_speed']:
                    ui['drone_speed'] = 5

                drone.speed = ui['drone_speed']
                drone.duration = 1
                drone.distance = 5 # ui['distance_ts']

            start = st.button('Start system')
            if start:
                ### TIM CODE
                # create unique folder for this run - to log and store data
                secondsSinceEpoch = time.time()
                timeObj = time.localtime(secondsSinceEpoch)
                timeStamp = '%d-%d-%d %d-%d-%d' % (
                timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
                drone.runPath = drone_name + '/runs/' + ui['user'] + ' ' + timeStamp
                os.mkdir(drone.runPath)
                drone.photosPath = os.path.join(drone.runPath, 'photos')
                os.mkdir(drone.photosPath)
                drone.logPath = os.path.join(drone.runPath, 'log')
                os.mkdir(drone.logPath)

                # set path (later will be replaced by a module)
                if drone_name == 'tello':
                    path = np.array([
                        [0, 0, 0]
                        , [0, -50, 0]
                        , [0, -250, 0]
                        , [0, -400, 0]
                    ])
                elif drone_name == 'unreal':
                    '''path = np.array([
                        [0, 0, 0]
                        ,[10, 0, 0]
                        ,[10, 0, 10]
                        ,[120, -40, 10]
                        ,[120, -80, 10]
                        ,[170, -70, -20]
                    ])'''
                    path = np.array([
                        [0, 0, 0]
                        , [40, 0, 0]
                        , [80, 0, 0]
                        , [120, 0, 0]
                        , [160, 0, 0]
                        , [200, 0, 0]
                    ])

                # connect to drone
                drone.connect()
                time.sleep(2)

                # make decisions
                sample_rate = 0  # make decision after this many seconds
                drone.takeOff()
                # drone.moveTo(0, 0, 100)
                args = {}
                args['timestep'] = 0
                args['drone'] = drone
                args['path'] = path
                args['nextPoint'] = path[1]
                args['lastPoint'] = path[0]
                args['startPoint'] = path[0]
                args['endPoint'] = path[-1]
                args['pathstep'] = 1
                args['nSteps'] = path.shape[0]
                args['lastDirection'] = ''
                args['visions'] = visions
                args['coefficients'] = coefficients
                args['objectiveEpsilon'] = 1
                print(drone.getState())

                while (True):
                    # stepthrough = input('Next Timestep?')

                    # move one timestep up
                    args['timestep'] += 1
                    args['timePath'] = os.path.join(drone.photosPath, str(args['timestep']))
                    os.mkdir(args['timePath'])

                    # take photos for this timestep
                    drone.takePictures(args['timePath'])

                    # transform vision modules
                    for vision in visions:
                        if vision in ui['activated_mod']:
                            args[vision + '_readPath'] = os.path.join(args['timePath'], 'Scene.png')
                            args[vision + '_writePath'] = os.path.join(args['timePath'], vision + '.png')
                            args[vision + '_rewardsPath'] = os.path.join(args['timePath'], vision + '_rewards.png')
                            visions[vision].transform(args[vision + '_readPath'], args[vision + '_writePath'])

                    # make decision
                    # args = decision.decide(args)
                    print('progress', args['progress'])

                    # wait for next time step
                    time.sleep(sample_rate)

                    # exit when reached end
                    if args['progress'] == 'goal':
                        stepthrough = input('Finished! Exit?')
                        if drone_name == 'tello':
                            drone.flip()
                        break

                # clean up
                drone.disconnect()
                print('buayyyyyeeeee')
                st.subheader('Run finished!')
                #TODO: implement image visualization
            ### end main screen


main()