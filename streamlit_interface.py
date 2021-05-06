# @Angelica
import time
import os
from PIL import Image
from vision import Vision
# import Drone, Depth, Segmentation, Decision, Aerial, Scene_Parser
import Depth, Decision
import Drone, Segmentation, Scene_Parser
import numpy as np
import streamlit as st
from streamlit import caching
from skimage import io


# start: module loaders: done this way for caching reasons
# @st.cache(hash_funcs={Aerial.Fire:id})
# def load_aerial_fire():
#     return Aerial.Fire()
#
# @st.cache(hash_funcs={Aerial.Building:id})
# def load_aerial_building():
#     return Aerial.Building()
#
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
    aerialFire_mod = st.sidebar.checkbox('Fire Segmentation')
    aerialObjects_mod = st.sidebar.checkbox('Smoke Segmentation')
    depth_mod = st.sidebar.checkbox('Depth Estimation')
    segmentation_mod = st.sidebar.checkbox('Object Detection')
    # scene_parse = st.sidebar.checkbox('Scene Parser')

    selected_mods = []
    if aerialFire_mod:
        selected_mods.append('aerialFire')
    if aerialObjects_mod:
        selected_mods.append('aerialObjects')
    if depth_mod:
        selected_mods.append('depth')
    if segmentation_mod:
        # selected_mods.append('segmentation')
        selected_mods.append('smoke')
    # if scene_parse:
    #     # selected_mods.append('segmentation')
    #     selected_mods.append('scene_parse')
    return selected_mods


def select_rl():
    st.sidebar.text('_______________________________________')
    st.sidebar.text('Select decision maker to use:')
    gir = st.sidebar.checkbox('Greedy Immediate Rewards')
    dql = st.sidebar.checkbox('Deep Q-Learning')
    fr = st.sidebar.checkbox('Free Roam')
    sp = st.sidebar.checkbox('Static Path')
    if gir:
        return 'Greedy Immediate Rewards'
    elif dql:
        return 'Deep Q-Learning'
    elif fr:
        return 'Free Roam'
    elif sp:
        return 'Static Path'


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

    # dict to contain all user inputs
    #aerial_fire = load_aerial_fire()
    #aerial_building = load_aerial_building()
    depth = load_depth()
    #segmentation = load_segmentation()
    #scene_parser = load_scene_parser()

    # define modules
    visions = {
        #'fire': aerial_fire,
        'depth': depth,
        #'smoke': segmentation,
        #'scene_parse': scene_parser
    }
    decision = Decision.Rewards_v3()

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

    #st.subheader(f'Hi {ui["user"]}, you are using {ui["drone_name"]}')

    # select mods
    if (ui['user'] != '') and (ui['drone_name']!=''):
        ui['activated_mod'] = select_mods()
        ui['decision_mods'] = select_rl()

        # only move forward to show further element after these elements are set
        ### start main screen
        #while len(ui['activated_mod'])==0:
        #    st.warning('Select at least one CV module')
        #    st.stop()


        #TODO: implement RL selection control

        
        #st.subheader('Flight parameters')
        #col1, col2 = st.beta_columns(2)
        #with col1:
        #    ui['distance_ts'] = st.text_input('Insert distance to cover per timestep')
        #    if not ui['distance_ts']: #TODO implement regex
        #        st.warning('This field cannot be null')
        #        st.stop()
        #with col2:
        #    ui['drone_speed'] = st.text_input('Insert drone speed', value=5)
        #    if not ui['drone_speed']:
        #        ui['drone_speed'] = 5
        #
        #    drone.speed = ui['drone_speed']
        #    drone.duration = 1
        #    drone.distance = 5 # ui['distance_ts']
        

        start = st.sidebar.button('Start Run')
        if start:
            #TODO: disable button during run
            # start.enabled = False
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
                #print(ui['activated_mod'])
                cols = vision_holder.beta_columns(len(visions))
                for vI, vision in enumerate(visions):
                    #if vision in ui['activated_mod']:
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
            #im_dir = args['timePath'] #os.path.join(drone_name, 'runs', ui['user'] + ' ' + timeStamp, 'photos', str(args['timestep']))
            drone.disconnect()
            print('buayyyyyeeeee')
            # time.sleep(2)
            st.subheader('----- Run finished! -----')
            # TODO: add map image
            # st.image()
            # ui['frame_to_view'] = st.text_input(f'Insert frame to view 0 to {args["timestep"]}')
            # if not ui['frame_to_view']:
            #     st.warning('insert a value from [0 to {')
            #     st.stop()

            # original view
            st.image(io.imread(os.path.join(im_dir, 'Scene.png')), caption='Original frame')

            res_smoke, res_fire = st.beta_columns(2)
            res_depth, res_parsing = st.beta_columns(2)

            smoke_img = os.path.join(im_dir,'smoke.png')
            if os.path.exists(smoke_img):
                with res_smoke:
                    st.image( io.imread(smoke_img),
                                caption='Smoke segmentation result',
                                use_column_width=True)

            depth_img = os.path.join(im_dir, 'depth.png' )
            if os.path.exists(depth_img):
                with res_depth:
                    st.image( io.imread(depth_img),
                                caption='Depth module result',
                                use_column_width=True)

            fire_img = os.path.join(im_dir, 'fire.png' )
            if os.path.exists(fire_img):
                with res_fire:
                    st.image( io.imread(os.path.join(im_dir, 'fire .png')),
                                caption='Fire module result',
                                use_column_width=True)

            scene_img = os.path.join(im_dir, 'scene_parse.png')
            if os.path.exists(scene_img):
                with res_parsing:
                    st.image(io.imread(os.path.join(im_dir, 'scene_parse.png')),
                                caption = 'Segmentation module result',
                                use_column_width=True)

            # TODO: enable start button after run is finished
            # start.enabled = True

        ### end main screen

main()