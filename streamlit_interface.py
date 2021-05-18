# @Angelica
import time
import os
from PIL import Image
from vision import Vision
import Aerial 
import Depth, Decision
import Drone, Segmentation, Scene_Parser
import numpy as np
import streamlit as st
from streamlit import caching
from skimage import io


# start: module loaders: done this way for caching reasons
@st.cache(hash_funcs={Aerial.Fire:id})
def load_aerial_fire():
    return Aerial.Fire()
@st.cache(hash_funcs={Aerial.Building:id})
def load_aerial_building():
    return Aerial.Building()
@st.cache(hash_funcs={Depth.MonoDepth2:id})
def load_depth():
    return Depth.MonoDepth2()

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
        drone.speed = 20
        drone.duration = 2
        drone.distance = 20
    elif drone_name == 'unreal':
        drone = Drone.Unreal()
        drone.speed = 5
        drone.duration = 1
        drone.distance = 5

    return drone, drone_name


def select_mods():
    st.sidebar.text('_________________________________')
    st.sidebar.text('Select vision modules to activate:')
    fire_mod = st.sidebar.checkbox('Fire Segmentation')
    smoke_mod = st.sidebar.checkbox('Smoke Segmentation')
    depth_mod = st.sidebar.checkbox('Depth Estimation')
    #parser_mod = st.sidebar.checkbox('Scene Parser')

    selected_mods = []
    if fire_mod:
        selected_mods.append('fire')
    if smoke_mod:
        selected_mods.append('smoke')
    if depth_mod:
        selected_mods.append('depth')
    #if parser_mod:
    #    selected_mods.append('parser')

    return selected_mods


def select_rl():
    st.sidebar.text('_______________________________________')
    st.sidebar.text('Select decision maker to use:')
    gir = st.sidebar.checkbox('Greedy Immediate Rewards')
    dql = st.sidebar.checkbox('Deep Q-Learning')
    fr = st.sidebar.checkbox('Free Roam')
    sp = st.sidebar.checkbox('Static Path')
    if gir:
        return 'greedy'
    elif dql:
        return 'deep'
    elif fr:
        return 'free'
    elif sp:
        return 'path'


def main():
    # RL coefficients
    coefficients = {
        'fire': 0,
        'depth': 1.5,
        'smoke': 0,
        'path': 1,
        'objective': 0,
        'smooth': 1
    }
    segmentation_slot = st.empty()

    # if aerial_fire and aerial_building and depth and segmentation and scene_parser:
    #if segmentation and scene_parser:
    # start ui
    ui = {}
    args = []
    #st.title('Computer Vision Aided Tello Observer')

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

    # select mods
    if (ui['user'] != '') and (ui['drone_name']!=''):
        ui['vision_mods'] = select_mods()
        ui['decision_mod'] = select_rl()


        start = st.sidebar.button('Start Run')
        if start:

            print(ui['vision_mods'])
            print(ui['decision_mod'])

            # set computer vision modules to view
            visions = {
            }
            if 'fire' in ui['vision_mods']:
                visions['fire'] = load_aerial_fire()
            if 'smoke' in ui['vision_mods']:
                visions['smoke'] = load_segmentation()
            if 'depth' in ui['vision_mods']:
                visions['depth'] = load_depth()
            if 'parser' in ui['vision_mods']:
                visions['parser'] = load_scene_parser()
    
            if ui['decision_mod'] == 'greedy':
                decision = Decision.Rewards_v3()
            if ui['decision_mod'] == 'free':
                decision = Decision.Input()
            if ui['decision_mod'] == 'deep':
                decision = Decision.Deep()
            if ui['decision_mod'] == 'path':
                decision = Decision.Path()

            print(ui['vision_mods'])
            print(ui['decision_mod'])

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
                # Demo 1
                #path = np.array([
                #    [0, 0, 0]
                #    , [0, -50, 0]
                #    , [0, -250, 0]
                #    , [0, -400, 0]
                #])
                # TV1 
                path = np.array([
                    [0, 0, 0]
                    , [80, 0, 0]
                    , [160, 0, 0]
                    , [240, 0, 0]
                    , [320, 0, 0]
                    , [400, 0, 0]
                ])
            elif drone_name == 'unreal':
                # Demo 1
                #path = np.array([
                #    [0, 0, 0]
                #    ,[10, 0, 0]
                #    ,[10, 0, 10]
                #    ,[120, -40, 10]
                #    ,[120, -80, 10]
                #    ,[170, -70, -20]
                #])
                # Moutain1
                # unreal pos = [47214.566406, 77266.492188, 21884.542969]
                path = np.array([
                    [0, 0, 100]
                    , [40, 0, 100]
                    , [80, 0, 100]
                    , [120, 0, 100]
                    , [160, 0, 100]
                    , [200, 0, 100]
                ])

            # connect to drone
            drone.connect()
            time.sleep(2)

            # make decisions
            wait_time = 0  # pause this many seconds after each decision
            drone.takeOff()
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
            
            scene_holder = st.empty()
            vision_holder = st.empty()
            rewards_holder = st.empty()
            while (True):
                # stepthrough = input('Next Timestep?')

                # move one timestep up
                args['timestep'] += 1
                args['timePath'] = os.path.join(drone.photosPath, str(args['timestep']))
                os.mkdir(args['timePath'])

                # take photos for this timestep
                drone.takePictures(args['timePath'])
                scene_holder.image(io.imread(os.path.join(args['timePath'], 'Scene.png')), caption='Scene Frame'
                            , width=330
                            )
                
                # transform vision modules and display
                cols = vision_holder.beta_columns(len(visions))
                for vI, vision in enumerate(visions):
                    args[vision + '_readPath'] = os.path.join(args['timePath'], 'Scene.png')
                    args[vision + '_writePath'] = os.path.join(args['timePath'], vision + '.png')
                    args[vision + '_rewardsPath'] = os.path.join(args['timePath'], vision + '_rewards.png')
                    visions[vision].transform(args[vision + '_readPath'], args[vision + '_writePath'])
                    with cols[vI]:
                        vision_holder.image( io.imread(args[vision + '_writePath']),
                                    caption= vision + ' Transformation',
                                    width=330
                                    #use_column_width=True
                                    )
                    #print(args[vision + '_writePath'])

                # make decision
                args = decision.decide(args)

                # display vision rewards
                cols = rewards_holder.beta_columns(len(visions))
                for vI, vision in enumerate(visions):
                    with cols[vI]:
                        rewards_holder.image( io.imread(args[vision + '_rewardsPath']),
                                    caption= vision + ' Rewards',
                                    width=330
                                    #use_column_width=True
                                    )
                    
                # pause before next timestep?
                time.sleep(wait_time)

                # exit when reached goal
                if args['progress'] == 'goal':
                    if drone_name == 'tello':
                        drone.flip()
                    break

            # clean up
            drone.disconnect()
            print('buayyyyyeeeee')
            st.subheader('----- Run finished! -----')

main()