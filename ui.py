import streamlit as st
import os
import time
#import Aerial, Drone, Depth, Segmentation, Decision
# import Drone, Depth,
import Drone, Segmentation
from skimage import io

@st.cache(hash_funcs={Segmentation.UNET: id})
def load_segmentation():
    print('Loading UNET')
    segmentation = Segmentation.UNET()
    print(type(segmentation))

    return segmentation


def select_mods():
    st.sidebar.text('Select modules to activate')
    aerialFire_mod = st.sidebar.checkbox('Aerial Fire')
    aerialObjects_mod = st.sidebar.checkbox('Aerial Objects')
    depth_mod = st.sidebar.checkbox('Depth')
    segmentation_mod = st.sidebar.checkbox('Segmentation')
    decision_mod = st.sidebar.checkbox('Decision', value=True)

    selected_mods = []
    if aerialFire_mod:
        selected_mods.append('aerialFire_mod')
    if aerialObjects_mod:
        selected_mods.append('aerialObjects_mod')
    if depth_mod:
        selected_mods.append('depth_mod')
    if segmentation_mod:
        selected_mods.append('segmentation_mod')
    if decision_mod:
        selected_mods.append('decision_mod')

    return selected_mods


def select_drone():
    drone_name = st.sidebar.text_input('Running from **Tello** or **Unreal**?')
    drone_names = ['Tello', 'Unreal']
    if drone_name not in drone_names:
        st.sidebar.warning('Please either Tello or Unreal')
        st.stop()

    drone_name = drone_name.lower()
    if drone_name == 'tello':
        drone = Drone.Tello()
    elif drone_name == 'unreal':
        drone = Drone.Unreal()
        drone.speed = 5

    return drone, drone_name


def start_ui(models):
    st.title('Computer Vision Aided Tello Observer')

    slots = []
    for i in range(0, 10):
        slots.append(st.empty())


    user = st.sidebar.selectbox('Select user:', ('', 'Angelica', 'Courtney', 'Olivia', 'Tim'))
    # select_mods(user)
    drone, drone_name = select_drone()
    st.subheader(f'Hi {user}, you are using {drone_name}')

    # select mods
    if user and drone_name:
        activated_mods = select_mods()


    st.subheader('GPS Parameters')
    a1, a2, a3 = st.beta_columns(3)
    with a1:
        gp1 = st.text_input('Start point: ')
    with a2:
        gp2 = st.text_input('End poi: ')
    with a3:
        gp3 = st.text_input('Speed: ')

    st.subheader('Flight Parameters')
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        iterations = st.text_input('Insert number of iterations: ')
        if not iterations:
            iterations = 5
    with col2:
        sample_rate = st.text_input('Wait time for decision: ')
        if not sample_rate:
            sample_rate = 0
    if drone_name == 'unreal':
        with col3:
            drone_speed = st.text_input('Drone speed: ')
            if not drone_speed:
                drone_speed = 5
    #define drone speed
    drone.speed = drone_speed

    go = st.button('Start system')
    if go:
        # TIM code
        # create unique folder for this run - to log and store data
        secondsSinceEpoch = time.time()
        timeObj = time.localtime(secondsSinceEpoch)
        timeStamp = '%d-%d-%d %d-%d-%d' % (
        timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
        drone.runPath = drone_name + '/runs/' + user + ' ' + timeStamp
        os.mkdir(drone.runPath)
        drone.photosPath = os.path.join(drone.runPath, 'photos')
        os.mkdir(drone.photosPath)
        drone.logPath = os.path.join(drone.runPath, 'log')
        os.mkdir(drone.logPath)

        # set path (later will be replaced by a module)
        args = {}
        if drone_name == 'tello':
            args['path'] = [
                [0, 0, 0]
                , [0, -50, 0]
                , [0, -250, 0]
                , [0, -400, 0]
            ]
        elif drone_name == 'unreal':
            args['path'] = [
                [0, 0, 0]
                , [10, 0, 0]
                , [10, 0, 10]
                , [120, -40, 10]
                , [120, -80, 10]
                , [170, -70, -20]
            ]

            # connect to drone
        timestep = 0
        drone.connect()
        time.sleep(2)
        args['framesPath'] = os.path.join(drone.photosPath, str('frames'))
        os.mkdir(args['framesPath'])
        # drone.liveStream({'MonoDepth2': depth, 'ColorFire': aerialFire, 'UNET': segmentation}, args['framesPath'])

        # snap aerial photos
        args['aerialPath'] = os.path.join(drone.photosPath, str(timestep))
        os.mkdir(args['aerialPath'])
        drone.snapAerial(args['aerialPath'])

        # get paths to aerial photos
        if drone_name == 'tello':
            args['aerialObjects_readPath'] = os.path.join(args['aerialPath'], 'SatelliteObjects.png')
            args['aerialFire_readPath'] = os.path.join(args['aerialPath'], 'SatelliteFire.png')
        if drone_name == 'unreal':
            args['aerialObjects_readPath'] = os.path.join(args['aerialPath'], 'Scene.png')
            args['aerialFire_readPath'] = os.path.join(args['aerialPath'], 'Scene.png')
        args['aerialObjects_writePath'] = os.path.join(args['aerialPath'], 'aerialObjects.png')
        args['aerialFire_writePath'] = os.path.join(args['aerialPath'], 'aerialFire.png')

        while (timestep < int(iterations)):
            # stepthrough = input('Next Timestep?')
            # stepthrough = st.text_input('Yes or no')
            # if 'y' in stepthrough.lower():
            #     stepthrough = True
            # else:
            #     stepthrough = False

            # move one timestep up
            timestep += 1
            print(timestep)
            args['timePath'] = os.path.join(drone.photosPath, str(timestep))
            os.mkdir(args['timePath'])

            # make decision
            # response = decision.decide(drone, args, timestep)

            # take photos for this timestep
            drone.takePictures(args['timePath'])

            # transform depth
            if 'depth_mod' in activated_mods:
                st.write('Depth module in action')
                args['depth_readPath'] = os.path.join(args['timePath'], 'Scene.png')
                args['depth_writePath'] = os.path.join(args['timePath'], 'depth.png')
                # depth.transform(args['depth_readPath'], args['depth_writePath'])

            # transform firesmoke segmentation
            if 'segmentation_mod' in activated_mods:
                # st.write('Segmentation in action')
                segmentation = models['segmentation']
                args['segmentation1_readPath'] = os.path.join(args['timePath'], 'Scene.png')
                args['segmentation1_writePath'] = os.path.join(args['timePath'], 'firesmoke_segmentation_1.png')
                segmentation.transform(args['segmentation1_readPath'], args['segmentation1_writePath'])

                # extra transform firesmoke segmentation (olivias)
                args['segmentation2_readPath'] = os.path.join(args['timePath'], 'Scene.png')
                args['segmentation2_writePath'] = os.path.join(args['timePath'], 'firesmoke_segmentation_2.png')
                # aerialFire.transform(args['segmentation2_readPath'], args['segmentation2_writePath'])

            # make decision
            # make decision
            # response = decision.decide(drone, args, timestep)

            # wait for next time step
            time.sleep(sample_rate)

            # exit when reached end
            # if response == 'quit':
            #     stepthrough = input('Finished! Exit?')
            #     if drone_name == 'tello':
            #         drone.flip()
            #     break

            # clean up
        drone.disconnect()
        print('buayyyyyeeeee')
        return ['Done', drone.photosPath]

def main():
    # set segmentaion module
    segmentation = load_segmentation()
    mods = {}
    if segmentation:

        # print('finished loading')
        mods['segmentation'] = segmentation

    start_ui(mods)


if __name__ == "__main__":
    main()
    #TODO: move elements in main
    #TODO: make image visualization
    #TODO: add restart button