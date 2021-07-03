# Wildfire_Drone_SJSU_2021
SJSU Spring 2021 Masters Project - Interface Drone Autonomy with Simulation and Real-World Environments

This open-source repo connects several computer vision models, satellite surveillance methods, and reinforcement learning methods for drone autonomy. 

It was built in Python and can interface with either a DJI Tello drone or Unreal Engine environment built with Microsoft AirSim.

Each component of this code is heavily modulated to rotate in/out different models and features, 
and can easily be adjusted to connect to any other drone or simulation environment.

We have a user interface built with streamlit.io, or the user can dive into the code to make simple edits.

To run the streamlit.io UI, launch from terminal: "streamlit run streamlit_interface.py"

The spreadsheet "Drone_Runs_log.xlsx" logs details about significant drone runs/trials we did.
See our Youtube playlist which catalogs videos and voice overs about each run:
https://www.youtube.com/playlist?list=PL68q5Piilhq70QzjtMliZv0Uw3WYOBorr

Within either the unreal or tello folder, the reader can find outputs for each run at each time step,
such as various computer vision transformations.

The current state of the streamlit_interface that is supported will set the drone on a short path forwrad,
and will use simple immediate greedy rewards and depth to try and avoid obsticales. 
Some editing is needed for more advanced methods.

Please download all of our large files from our google drive, and place in the parent folder of your directory with github code.
https://drive.google.com/drive/folders/1bpHiVPMtAaCCSrJbe5Co1lcWYeROO-DC?usp=sharing
