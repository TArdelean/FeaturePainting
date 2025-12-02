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
