import streamlit as st
import os
import time
#import Aerial, Drone, Depth, Segmentation, Decision
import Drone, Depth, Segmentation
#TODO: add thread to load modules
#TODO: add post processing



st.markdown('Computer Vision Aided Tello Observer')

# get user Info
user = st.selectbox('Select User',
                    ['', 'Angelica', 'Courtney','Olivia','Tim'])
# Drone.hello()

#set all modules
# set aerial fire module
#aerialFire = Aerial.Fire()

# set aerial object module
#aerialObjects = Aerial.Objects()

# set depth module
depth = Depth.MonoDepth2()

# set segmentaion module
segmentation = Segmentation.UNET()

# set decision module
# decision = Decision.Input()
# decision = Decision.Path()
# decision = Decision.Deep()


if user != '':
    st.write('Hello ', user, '!')
    st.write('===> Drone', Drone.hello())

    # select all modules to activate
    # streamlit.checkbox(label, value=False, key=None, help=None)
    st.text('Select modules to activate')
    aerialFire_mod = st.checkbox('Aerial Fire')
    aerialObjects_mod = st.checkbox('Aerial Objects')
    depth_mod = st.checkbox('Depth')
    segmentation_mod = st.checkbox('Segmentation')
    decision_mod = st.checkbox('Decision', value=True)

    drone_name = st.text_input('Running from **Tello** or **Unreal**?')
    drone_names = ['Tello', 'Unreal']
    if drone_name not in drone_names:
        st.warning('Please either Tello or Unreal')
        st.stop()
    else:
        st.text('Activating modules .... ')

    drone_name = drone_name.lower()
    if drone_name == 'tello':
        drone = Drone.Tello()
    elif drone_name == 'unreal':
        drone = Drone.Unreal()
        drone.speed = 5

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

    # transform aerial photos
    #if aerialObjects_mod:
        #aerialObjects = Aerial.Objects()
        # aerialObjects.transform(args['aerialObjects_readPath'], args['aerialObjects_writePath'])
    #if aerialFire:
        #aerialFire = Aerial.Fire()
        # aerialFire.transform(args['aerialFire_readPath'], args['aerialFire_writePath'])

    # make decisions
    sample_rate = 0  # make decision after this many seconds
    drone.takeOff()
    while (True):
        stepthrough = input('Next Timestep?')

        # move one timestep up
        timestep += 1
        args['timePath'] = os.path.join(drone.photosPath, str(timestep))
        os.mkdir(args['timePath'])

        # make decision
        # response = decision.decide(drone, args, timestep)

        # take photos for this timestep
        drone.takePictures(args['timePath'])

        # transform depth
        if depth_mod:
            st.write('Depth module in action')
            args['depth_readPath'] = os.path.join(args['timePath'], 'Scene.png')
            args['depth_writePath'] = os.path.join(args['timePath'], 'depth.png')
            depth.transform(args['depth_readPath'], args['depth_writePath'])

        # transform firesmoke segmentation
        if segmentation_mod:
            st.write('Segmentation in action')
            args['segmentation1_readPath'] = os.path.join(args['timePath'], 'Scene.png')
            args['segmentation1_writePath'] = os.path.join(args['timePath'], 'firesmoke_segmentation_1.png')
            segmentation.transform(args['segmentation1_readPath'], args['segmentation1_writePath'])

            # extra transform firesmoke segmentation (olivias)
            args['segmentation2_readPath'] = os.path.join(args['timePath'], 'Scene.png')
            args['segmentation2_writePath'] = os.path.join(args['timePath'], 'firesmoke_segmentation_2.png')
            # aerialFire.transform(args['segmentation2_readPath'], args['segmentation2_writePath'])

        #make decision
        # make decision
        #response = decision.decide(drone, args, timestep)

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

post_process = st.button('Post process images')
if post_process:
    #get images
    pass