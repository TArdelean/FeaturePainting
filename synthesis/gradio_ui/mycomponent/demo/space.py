
import gradio as gr
from app import demo as app
import os

_docs = {'MyComponent': {'description': 'Creates an image component that, as an input, can be used to upload and edit images using simple editing tools such\nas brushes, strokes, cropping, and layers. Or, as an output, this component can be used to display images.\n', 'members': {'__init__': {'value': {'type': 'EditorValue | numpy.ndarray | PIL.Image.Image | str | None', 'default': 'None', 'description': 'Optional initial image(s) to populate the image editor. Should be a dictionary with keys: `background`, `layers`, and `composite`. The values corresponding to `background` and `composite` should be images or None, while `layers` should be a list of images. Images can be of type PIL.Image, np.array, or str filepath/URL. Or, the value can be a callable, in which case the function will be called whenever the app loads to set the initial value of the component.'}, 'height': {'type': 'int | str | None', 'default': 'None', 'description': 'The height of the displayed images, specified in pixels if a number is passed, or in CSS units if a string is passed.'}, 'width': {'type': 'int | str | None', 'default': 'None', 'description': 'The width of the displayed images, specified in pixels if a number is passed, or in CSS units if a string is passed.'}, 'image_mode': {'type': '"1"\n    | "L"\n    | "P"\n    | "RGB"\n    | "RGBA"\n    | "CMYK"\n    | "YCbCr"\n    | "LAB"\n    | "HSV"\n    | "I"\n    | "F"', 'default': '"RGBA"', 'description': '"RGB" if color, or "L" if black and white. See https://pillow.readthedocs.io/en/stable/handbook/concepts.html for other supported image modes and their meaning.'}, 'sources': {'type': 'Iterable["upload" | "webcam" | "clipboard"]', 'default': '"upload", "webcam", "clipboard"', 'description': 'List of sources that can be used to set the background image. "upload" creates a box where user can drop an image file, "webcam" allows user to take snapshot from their webcam, "clipboard" allows users to paste an image from the clipboard.'}, 'type': {'type': '"numpy" | "pil" | "filepath"', 'default': '"numpy"', 'description': 'The format the images are converted to before being passed into the prediction function. "numpy" converts the images to numpy arrays with shape (height, width, 3) and values from 0 to 255, "pil" converts the images to PIL image objects, "filepath" passes images as str filepaths to temporary copies of the images.'}, 'label': {'type': 'str | None', 'default': 'None', 'description': 'The label for this component. Appears above the component and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component is assigned to.'}, 'every': {'type': 'float | None', 'default': 'None', 'description': "If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute."}, 'show_label': {'type': 'bool | None', 'default': 'None', 'description': 'if True, will display label.'}, 'show_download_button': {'type': 'bool', 'default': 'True', 'description': 'If True, will display button to download image.'}, 'container': {'type': 'bool', 'default': 'True', 'description': 'If True, will place the component in a container - providing some extra padding around the border.'}, 'scale': {'type': 'int | None', 'default': 'None', 'description': 'relative size compared to adjacent Components. For example if Components A and B are in a Row, and A has scale=2, and B has scale=1, A will be twice as wide as B. Should be an integer. scale applies in Rows, and to top-level Components in Blocks where fill_height=True.'}, 'min_width': {'type': 'int', 'default': '160', 'description': 'minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.'}, 'interactive': {'type': 'bool | None', 'default': 'None', 'description': 'if True, will allow users to upload and edit an image; if False, can only be used to display images. If not provided, this is inferred based on whether the component is used as an input or output.'}, 'visible': {'type': 'bool', 'default': 'True', 'description': 'If False, component will be hidden.'}, 'elem_id': {'type': 'str | None', 'default': 'None', 'description': 'An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.'}, 'elem_classes': {'type': 'list[str] | str | None', 'default': 'None', 'description': 'An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.'}, 'render': {'type': 'bool', 'default': 'True', 'description': 'If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.'}, 'mirror_webcam': {'type': 'bool', 'default': 'True', 'description': 'If True webcam will be mirrored. Default is True.'}, 'show_share_button': {'type': 'bool | None', 'default': 'None', 'description': 'If True, will show a share icon in the corner of the component that allows user to share outputs to Hugging Face Spaces Discussions. If False, icon does not appear. If set to None (default behavior), then the icon appears if this Gradio app is launched on Spaces, but not otherwise.'}, '_selectable': {'type': 'bool', 'default': 'False', 'description': None}, 'crop_size': {'type': 'tuple[int | float, int | float] | str | None', 'default': 'None', 'description': 'The size of the crop box in pixels. If a tuple, the first value is the width and the second value is the height. If a string, the value must be a ratio in the form `width:height` (e.g. "16:9").'}, 'transforms': {'type': 'Iterable["crop"]', 'default': '"crop",', 'description': 'The transforms tools to make available to users. "crop" allows the user to crop the image.'}, 'eraser': {'type': 'Eraser | None | False', 'default': 'None', 'description': 'The options for the eraser tool in the image editor. Should be an instance of the `gr.Eraser` class, or None to use the default settings. Can also be False to hide the eraser tool.'}, 'brush': {'type': 'Brush | None | False', 'default': 'None', 'description': 'The options for the brush tool in the image editor. Should be an instance of the `gr.Brush` class, or None to use the default settings. Can also be False to hide the brush tool, which will also hide the eraser tool.'}, 'format': {'type': 'str', 'default': '"webp"', 'description': 'Format to save image if it does not already have a valid format (e.g. if the image is being returned to the frontend as a numpy array or PIL Image).  The format should be supported by the PIL library. This parameter has no effect on SVG files.'}, 'layers': {'type': 'bool', 'default': 'True', 'description': 'If True, will allow users to add layers to the image. If False, the layers option will be hidden.'}}, 'postprocess': {'value': {'type': 'EditorValue | numpy.ndarray | PIL.Image.Image | str | None', 'description': "Expects a EditorValue, which is just a dictionary with keys: 'background', 'layers', and 'composite'. The values corresponding to 'background' and 'composite' should be images or None, while `layers` should be a list of images. Images can be of type `PIL.Image`, `np.array`, or `str` filepath/URL. Or, the value can be simply a single image (`ImageType`), in which case it will be used as the background."}}, 'preprocess': {'return': {'type': 'EditorValue | None', 'description': "Passes the uploaded images as an instance of EditorValue, which is just a `dict` with keys: 'background', 'layers', and 'composite'. The values corresponding to 'background' and 'composite' are images, while 'layers' is a `list` of images. The images are of type `PIL.Image`, `np.array`, or `str` filepath, depending on the `type` parameter."}, 'value': None}}, 'events': {'clear': {'type': None, 'default': None, 'description': 'This listener is triggered when the user clears the MyComponent using the X button for the component.'}, 'change': {'type': None, 'default': None, 'description': 'Triggered when the value of the MyComponent changes either because of user input (e.g. a user types in a textbox) OR because of a function update (e.g. an image receives a value from the output of an event trigger). See `.input()` for a listener that is only triggered by user input.'}, 'input': {'type': None, 'default': None, 'description': 'This listener is triggered when the user changes the value of the MyComponent.'}, 'select': {'type': None, 'default': None, 'description': 'Event listener for when the user selects or deselects the MyComponent. Uses event data gradio.SelectData to carry `value` referring to the label of the MyComponent, and `selected` to refer to state of the MyComponent. See EventData documentation on how to use this event data'}, 'upload': {'type': None, 'default': None, 'description': 'This listener is triggered when the user uploads a file into the MyComponent.'}, 'apply': {'type': None, 'default': None, 'description': 'This listener is triggered when the user applies changes to the MyComponent through an integrated UI action.'}}}, '__meta__': {'additional_interfaces': {'EditorValue': {'source': 'class EditorValue(TypedDict):\n    background: Optional[ImageType]\n    layers: list[ImageType]\n    composite: Optional[ImageType]'}, 'Eraser': {'source': '@dataclasses.dataclass\nclass Eraser:\n    default_size: int | Literal["auto"] = "auto"'}, 'Brush': {'source': '@dataclasses.dataclass\nclass Brush(Eraser):\n    colors: Union[\n        list[str],\n        str,\n        None,\n    ] = None\n    default_color: Union[str, Literal["auto"]] = "auto"\n    color_mode: Literal["fixed", "defaults"] = "defaults"\n\n    def __post_init__(self):\n        if self.colors is None:\n            self.colors = [\n                "rgb(204, 50, 50)",\n                "rgb(173, 204, 50)",\n                "rgb(50, 204, 112)",\n                "rgb(50, 112, 204)",\n                "rgb(173, 50, 204)",\n            ]\n        if self.default_color is None:\n            self.default_color = (\n                self.colors[0]\n                if isinstance(self.colors, list)\n                else self.colors\n            )'}}, 'user_fn_refs': {'MyComponent': ['EditorValue']}}}

abs_path = os.path.join(os.path.dirname(__file__), "css.css")

with gr.Blocks(
    css=abs_path,
    theme=gr.themes.Default(
        font_mono=[
            gr.themes.GoogleFont("Inconsolata"),
            "monospace",
        ],
    ),
) as demo:
    gr.Markdown(
"""
# `gradio_mycomponent`

<div style="display: flex; gap: 7px;">
<a href="https://pypi.org/project/gradio_mycomponent/" target="_blank"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/gradio_mycomponent"></a>  
</div>

my image editor
""", elem_classes=["md-custom"], header_links=True)
    app.render()
    gr.Markdown(
"""
## Installation

```bash
pip install gradio_mycomponent
```

## Usage

```python
import gradio as gr
from gradio_mycomponent import MyComponent
from gradio_mycomponent.mycomponent import Brush


def rgb_to_hex(rgb):
    return "#" + '%02x%02x%02x' % tuple(rgb)


colors = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 255], [0, 255, 0], [0, 128, 128]]
colors_hex = [rgb_to_hex(c) for c in colors]

# example = MyComponent().example_value()
editor = MyComponent(sources=['upload'],
                     brush=Brush(colors=colors_hex[1:], color_mode='fixed'),
                     crop_size="8:8",
                     image_mode="RGB",
                     )
output_frame = MyComponent()


def some_fun(x):
    return x


demo = gr.Interface(
    some_fun,
    editor,  # interactive version of your component
    output_frame,  # static version of your component
    # examples=[[example]],  # uncomment this line to view the "example version" of your component
)

if __name__ == "__main__":
    demo.launch()

```
""", elem_classes=["md-custom"], header_links=True)


    gr.Markdown("""
## `MyComponent`

### Initialization
""", elem_classes=["md-custom"], header_links=True)

    gr.ParamViewer(value=_docs["MyComponent"]["members"]["__init__"], linkify=['EditorValue', 'Eraser', 'Brush'])


    gr.Markdown("### Events")
    gr.ParamViewer(value=_docs["MyComponent"]["events"], linkify=['Event'])




    gr.Markdown("""

### User function

The impact on the users predict function varies depending on whether the component is used as an input or output for an event (or both).

- When used as an Input, the component only impacts the input signature of the user function.
- When used as an output, the component only impacts the return signature of the user function.

The code snippet below is accurate in cases where the component is used as both an input and an output.

- **As input:** Is passed, passes the uploaded images as an instance of EditorValue, which is just a `dict` with keys: 'background', 'layers', and 'composite'. The values corresponding to 'background' and 'composite' are images, while 'layers' is a `list` of images. The images are of type `PIL.Image`, `np.array`, or `str` filepath, depending on the `type` parameter.
- **As output:** Should return, expects a EditorValue, which is just a dictionary with keys: 'background', 'layers', and 'composite'. The values corresponding to 'background' and 'composite' should be images or None, while `layers` should be a list of images. Images can be of type `PIL.Image`, `np.array`, or `str` filepath/URL. Or, the value can be simply a single image (`ImageType`), in which case it will be used as the background.

 ```python
def predict(
    value: EditorValue | None
) -> EditorValue | numpy.ndarray | PIL.Image.Image | str | None:
    return value
```
""", elem_classes=["md-custom", "MyComponent-user-fn"], header_links=True)




    code_EditorValue = gr.Markdown("""
## `EditorValue`
```python
class EditorValue(TypedDict):
    background: Optional[ImageType]
    layers: list[ImageType]
    composite: Optional[ImageType]
```""", elem_classes=["md-custom", "EditorValue"], header_links=True)

    code_Eraser = gr.Markdown("""
## `Eraser`
```python
@dataclasses.dataclass
class Eraser:
    default_size: int | Literal["auto"] = "auto"
```""", elem_classes=["md-custom", "Eraser"], header_links=True)

    code_Brush = gr.Markdown("""
## `Brush`
```python
@dataclasses.dataclass
class Brush(Eraser):
    colors: Union[
        list[str],
        str,
        None,
    ] = None
    default_color: Union[str, Literal["auto"]] = "auto"
    color_mode: Literal["fixed", "defaults"] = "defaults"

    def __post_init__(self):
        if self.colors is None:
            self.colors = [
                "rgb(204, 50, 50)",
                "rgb(173, 204, 50)",
                "rgb(50, 204, 112)",
                "rgb(50, 112, 204)",
                "rgb(173, 50, 204)",
            ]
        if self.default_color is None:
            self.default_color = (
                self.colors[0]
                if isinstance(self.colors, list)
                else self.colors
            )
```""", elem_classes=["md-custom", "Brush"], header_links=True)

    demo.load(None, js=r"""function() {
    const refs = {
            EditorValue: [], 
            Eraser: [], 
            Brush: [], };
    const user_fn_refs = {
          MyComponent: ['EditorValue'], };
    requestAnimationFrame(() => {

        Object.entries(user_fn_refs).forEach(([key, refs]) => {
            if (refs.length > 0) {
                const el = document.querySelector(`.${key}-user-fn`);
                if (!el) return;
                refs.forEach(ref => {
                    el.innerHTML = el.innerHTML.replace(
                        new RegExp("\\b"+ref+"\\b", "g"),
                        `<a href="#h-${ref.toLowerCase()}">${ref}</a>`
                    );
                })
            }
        })

        Object.entries(refs).forEach(([key, refs]) => {
            if (refs.length > 0) {
                const el = document.querySelector(`.${key}`);
                if (!el) return;
                refs.forEach(ref => {
                    el.innerHTML = el.innerHTML.replace(
                        new RegExp("\\b"+ref+"\\b", "g"),
                        `<a href="#h-${ref.toLowerCase()}">${ref}</a>`
                    );
                })
            }
        })
    })
}

""")

demo.launch()
