# Stone_Soup_GMPHD
This repository implements a *concurrent search-and-track* guidance algorithm for a swarm of UAVs with multiple moving 
targets.

Specifically, the code implements a simulation of a swarm of UAV drones searching an area and attempting to track 
multiple moving t


argets.

<div><video controls src="https://user-images.githubusercontent.com/19653313/197231667-4d7014f2-36ec-4823-a2c0-980a16f4068e.mp4
" muted="false"></video></div>

*Figure 1: Output video of simulation. Checkout "Background" section for theoretical details of what's going on.*

The implementation of the guidance algorithm relies on the Python package 
"[Stone Soup](https://stonesoup.readthedocs.io/en/v0.1b11/)".

The ultimate goal is to integrate this source code into a UAV Autonomy Package.

## Usage
Read this section if you want to run your own simulations or modify code.

### Requirements
- Python (>= 3.9)
- FFmpeg (latest version)

### Installing Requirements
#### Mac
Homebrew recommended (N.B. homebrew install directory must be ```/opt/homebrew```):
```shell
brew install python@3.10 FFmpeg
```
#### Ubuntu
```shell
sudo apt-get update && sudo apt-get install ffmpeg python3
```

### Setup
1. Clone repo
```shell
git clone git@github.com:zborffs/Stone_Soup_GMPHD.git
```

2. Create Python virtual environment
```shell
python -m venv venv
```

3. Source virtual environment
```shell
source venv/bin/activate
```

4. Install required Python packages to virtual environment:
```shell
pip install -r requirements.txt
```

5. Run app
```shell
python main_guided.py 
```

**Expected output:**
An application window should get created that looks like this:
![Expected Initial App Window](figures/Initial_App_Window.png)

Select the window and press any key on your keyboard to continue the program.

You should see this in the terminal:
```shell
$ python main_guided.py 
--- Took 0.008393049240112305 seconds to initilize ---
--- Took 0.07797729583333333 seconds on average per iteration ---
--- Estimated Rate 12.824245689891251 Hz
Completed
```
A file called "output.mp4" should have been created. This video file is an visualization of the simulation. Note: It 
takes about 20 minutes to construct the video file. Comment out the line ```anim.save("output.mp4")```, and you should 
still be able to see the animation, it just won't save. 


## Background
This repository implements a guidance algorithm for a swarm of UAVs for multi

### Stonesoup Tutorials

## Resources
[1] Tal Shima, Steven Rasmussen, *UAV Cooperative Decision and Control: Challenges and Practical Approaches*, 2009
- Great resource for providing high-level background. A bit old.

[2] Stone Soup contributors (2022), *Stone Soup Read the Docs*, https://stonesoup.readthedocs.io/en/v0.1b11/
- Contains documentation of functions, classes, etc.. Has great tutorials.
