from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
devices = graph.get_input_devices()

print("Available camera devices:")
for i, device in enumerate(devices):
    print(f"{i}: {device}")
