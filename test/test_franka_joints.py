import genesis as gs
gs.init(backend=gs._gs_backend.cpu)
scene = gs.Scene(show_viewer=False)
robot = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
scene.build()
for joint in robot.joints:
    if joint.dofs_limit[0] is not None:
        print(f"Name: {joint.name}, Type: {joint.type}, Limits: {joint.dofs_limit}")
