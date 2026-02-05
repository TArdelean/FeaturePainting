# 🎨 FeaturePainting Blender Add-on

This add-on makes it easy to try FeaturePainting directly on 3D meshes in Blender.

## 🚀 Installation
1. Download the whole repository and set up the python environment (see the main [README](../README.md)).

2. Open Blender and navigate to Edit > Preferences > Add-ons.

3. Click 'Install from disk' at the top right and select blender_script.py.

4. Enable the add-on by checking the box next to "Texture Synthesis Add-on"

## ⚙️ Configuration

Before you can start painting, you need to link the add-on to the main project:

1. In the Add-on Preferences (where you just enabled it), find the Project Location field.

2. Select the root directory of your local FeaturePainting repository.

3. Note: This ensures the script can locate the necessary generation scripts and models.

## 🛠 How to Use
##### 1. Access the Panel

Press N in the 3D Viewport to open the Sidebar. Look for the FeaturePainting tab.
##### 2. Select Your Weights

Use the dropdown menu within the panel to select your desired model weights (Texture). These are pulled directly from the root project.
##### 3. Generate & Paint

Generate: Click the generation button to create your base texture.
   * Tip: Make sure you are in material preview mode (viewport shading) to see the textures

Paint: Once the texture is loaded, use the integrated Anomaly Brushes to paint features on your 3D object.
   * Tip: Ensure your object is properly UV-unwrapped before painting to avoid stretching!