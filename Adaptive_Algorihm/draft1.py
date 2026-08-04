class GraphNode:

    def __init__(self,x,y,z):

        self.x=x
        self.y=y
        self.z=z

        self.temperature=25.0

        self.level=0

        self.active=True

        self.surface=False

class ThermalGraph:

    def __init__(self):

        self.nodes=[]

        self.edges=[]

        self.L=None


graph.generate_uniform_grid(dx=1.0)

laser_position=np.array([x,y,z])

radius=3*beam_radius

near_nodes=[]

for node in graph.nodes:

    d=np.linalg.norm(
        laser_position-
        np.array([node.x,node.y,node.z])
    )

    if d<radius:

        near_nodes.append(node)