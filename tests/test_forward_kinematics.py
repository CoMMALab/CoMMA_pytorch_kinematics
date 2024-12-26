import math
import os
from timeit import timeit

import torch

import pytorch_kinematics as pk
from pytorch_kinematics import FKSolution 
fk = FKSolution()
from python_robotics_middleware.transforms.math import quaternion_close

TEST_DIR = os.path.dirname(__file__)
UM_ARM_LAB_ASSETS = os.path.join(TEST_DIR, "UM-ARM-Lab-Assets")


def quat_pos_from_transform3d(tg):
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot


# test more complex robot and the MJCF parser
def test_fk_mjcf():
    print("Test forward kinematics for MJCF robot")
    chain = pk.build_chain_from_mjcf(open(os.path.join(UM_ARM_LAB_ASSETS, "ant.xml")).read())
    chain = chain.to(dtype=torch.float64)
    print("\tRobot chain:", chain)
    print("\tJoint parameter names:", chain.get_joint_parameter_names())

    th = {joint: 0.0 for joint in chain.get_joint_parameter_names()}
    th.update({'hip_1': 1.0, 'ankle_1': 1})
    print("\tJoint angles:", th)

    ret = fk.forward_kinematics(chain, th)
    tg = ret['aux_1']
    pos, rot = quat_pos_from_transform3d(tg)

    if (quaternion_close(rot, torch.tensor([0.87758256, 0., 0., 0.47942554], dtype=torch.float64))):
        print("\tRotation Quaternions aux_1 match.")
    assert quaternion_close(rot, torch.tensor([0.87758256, 0., 0., 0.47942554], dtype=torch.float64))
    if torch.allclose(pos, torch.tensor([0.2, 0.2, 0.75], dtype=torch.float64)):
        print("\tPosition Quaternions aux_1 match.")
    assert torch.allclose(pos, torch.tensor([0.2, 0.2, 0.75], dtype=torch.float64))

    tg = ret['front_left_foot']
    pos, rot = quat_pos_from_transform3d(tg)
    if quaternion_close(rot, torch.tensor([0.77015115, -0.4600326, 0.13497724, 0.42073549], dtype=torch.float64)):
        print("\tRotation Quaternions front_left_foot match.")
    assert quaternion_close(rot, torch.tensor([0.77015115, -0.4600326, 0.13497724, 0.42073549], dtype=torch.float64))
    if torch.allclose(pos, torch.tensor([0.13976626, 0.47635466, 0.75], dtype=torch.float64)):
        print("\tPosition Quaternions front_left_foot match.")
    assert torch.allclose(pos, torch.tensor([0.13976626, 0.47635466, 0.75], dtype=torch.float64))
    print("\tForward kinematics test for MJCF completed successfully.")
    print(ret)


def test_fk_serial_mjcf():
    chain = pk.build_serial_chain_from_mjcf(open(os.path.join(UM_ARM_LAB_ASSETS, "ant.xml")).read(), 'front_left_foot')
    chain = chain.to(dtype=torch.float64)
    tg = fk.forward_kinematics(chain, [1.0, 1.0])
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_close(rot, torch.tensor([0.77015115, -0.4600326, 0.13497724, 0.42073549], dtype=torch.float64))
    assert torch.allclose(pos, torch.tensor([0.13976626, 0.47635466, 0.75], dtype=torch.float64))


def test_fkik():
    print("Test forward kinematics setup for inverse kinematics")
    data = '<robot name="test_robot">' \
           '<link name="link1" />' \
           '<link name="link2" />' \
           '<link name="link3" />' \
           '<joint name="joint1" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link1"/>' \
           '<child link="link2"/>' \
           '</joint>' \
           '<joint name="joint2" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link2"/>' \
           '<child link="link3"/>' \
           '</joint>' \
           '</robot>'
    chain = pk.build_serial_chain_from_urdf(data, 'link3')
    th1 = torch.tensor([0.42553542, 0.17529176])
    tg = fk.forward_kinematics(chain, th1)
    pos, rot = quat_pos_from_transform3d(tg)
    if quaternion_close(rot, torch.tensor([[0.95521418, 0.0000, 0.0000, 0.2959153]])):
        print("\tRotation Quaternion matches the expected value.")
    assert quaternion_close(rot, torch.tensor([[0.95521418, 0.0000, 0.0000, 0.2959153]]))
    if torch.allclose(pos, torch.tensor([[1.91081784, 0.41280851, 0.0000]])):
        print("\tPosition Quaternion matches the expected value.")
    assert torch.allclose(pos, torch.tensor([[1.91081784, 0.41280851, 0.0000]]))
    
    N = 20
    th_batch = torch.rand(N, 2)
    tg_batch = fk.forward_kinematics(chain, th_batch)
    m = tg_batch.get_matrix()
    for i in range(N):
        tg = fk.forward_kinematics(chain, th_batch[i])
        assert torch.allclose(tg.get_matrix().view(4, 4), m[i])

    # check that gradients are passed through
    th2 = torch.tensor([0.42553542, 0.17529176], requires_grad=True)
    tg = fk.forward_kinematics(chain,th2)
    pos, rot = quat_pos_from_transform3d(tg)
    # note that since we are using existing operations we are not checking grad calculation correctness
    assert th2.grad is None
    pos.norm().backward()
    assert th2.grad is not None

    print("\tGradient extraction is functional.")


def test_urdf():
    chain = pk.build_chain_from_urdf(open(os.path.join(UM_ARM_LAB_ASSETS, "kuka_iiwa.urdf")).read())
    chain.to(dtype=torch.float64)
    th = [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]
    ret = fk.forward_kinematics(chain,th)
    tg = ret['lbr_iiwa_link_7']
    pos, rot = quat_pos_from_transform3d(tg)
    if quaternion_close(rot, torch.tensor([7.07106781e-01, 0, -7.07106781e-01, 0], dtype=torch.float64)):
        print("\tRotation Quaternion matches the expected value.")
    assert quaternion_close(rot, torch.tensor([7.07106781e-01, 0, -7.07106781e-01, 0], dtype=torch.float64))
    if torch.allclose(pos, torch.tensor([-6.60827561e-01, 0, 3.74142136e-01], dtype=torch.float64), atol=1e-6):
        print("\tPosition Quaternion matches the expected value.")  
    assert torch.allclose(pos, torch.tensor([-6.60827561e-01, 0, 3.74142136e-01], dtype=torch.float64), atol=1e-6)


def test_urdf_serial():
    print("Test building serial chain from URDF")
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(UM_ARM_LAB_ASSETS, "kuka_iiwa.urdf")).read(), "lbr_iiwa_link_7")
    chain.to(dtype=torch.float64)
    th = [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]

    ret = fk.forward_kinematics(chain,th, end_only=False)
    tg = ret['lbr_iiwa_link_7']
    pos, rot = quat_pos_from_transform3d(tg)
    if quaternion_close(rot, torch.tensor([7.07106781e-01, 0, -7.07106781e-01, 0], dtype=torch.float64)):
        print("\tRotation Quaternion matches the expected value.")
    assert quaternion_close(rot, torch.tensor([7.07106781e-01, 0, -7.07106781e-01, 0], dtype=torch.float64))
    if torch.allclose(pos, torch.tensor([-6.60827561e-01, 0, 3.74142136e-01], dtype=torch.float64), atol=1e-6):
        print("\tPosition Quaternion matches the expected value.")
    assert torch.allclose(pos, torch.tensor([-6.60827561e-01, 0, 3.74142136e-01], dtype=torch.float64), atol=1e-6)

    N = 100
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    th_batch = torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)

    chain = chain.to(dtype=dtype, device=d)

    # NOTE: Warmstart since pytorch can be slow the first time you run it
    #  this has to be done after you move it to the GPU. Otherwise the timing isn't representative.
    print("Warm Starting GPU")
    for _ in range(5):
        ret = fk.forward_kinematics(chain,th)

    number = 10

    print("Test forward kinematics time on gpu using serial chain")
    def _fk_parallel():
        tg_batch = fk.forward_kinematics(chain,th_batch)
        m = tg_batch.get_matrix()

    dt_parallel = timeit(_fk_parallel, number=number) / number
    print("\telapsed {}s for N={} when parallel".format(dt_parallel, N))

    def _fk_serial():
        for i in range(N):
            tg = fk.forward_kinematics(chain,th_batch[i])
            m = tg.get_matrix()

    dt_serial = timeit(_fk_serial, number=number) / number
    print("\telapsed {}s for N={} when serial".format(dt_serial, N))

    # assert torch.allclose(tg.get_matrix().view(4, 4), m[i])


# test robot with prismatic and fixed joints
def test_fk_simple_arm():
    print("Test forward kinematics for simple arm")
    chain = pk.build_chain_from_sdf(open(os.path.join(UM_ARM_LAB_ASSETS, "simple_arm.sdf")).read())
    chain = chain.to(dtype=torch.float64)
    # print(chain)
    # print(chain.get_joint_parameter_names())
    ret = fk.forward_kinematics(chain,{
        'arm_shoulder_pan_joint': 0.,
        'arm_elbow_pan_joint': math.pi / 2.0,
        'arm_wrist_lift_joint': -0.5,
        'arm_wrist_roll_joint': 0.,
    })
    tg = ret['arm_wrist_roll']
    pos, rot = quat_pos_from_transform3d(tg)
    if quaternion_close(rot, torch.tensor([0.70710678, 0., 0., 0.70710678], dtype=torch.float64)):
        print("\tRotation Quaternion matches the expected value.")
    assert quaternion_close(rot, torch.tensor([0.70710678, 0., 0., 0.70710678], dtype=torch.float64))
    if torch.allclose(pos, torch.tensor([1.05, 0.55, 0.5], dtype=torch.float64)):
        print("\tPosition Quaternion matches the expected value.")
    assert torch.allclose(pos, torch.tensor([1.05, 0.55, 0.5], dtype=torch.float64))

    
    

    N = 100
    ret = fk.forward_kinematics(chain,{k: torch.rand(N) for k in chain.get_joint_parameter_names()})
    tg = ret['arm_wrist_roll']
    if list(tg.get_matrix().shape) == [N, 4, 4]:
        print("\tarm_wrist_roll kinematics has the correct dimensionality: [N,4,4]")
    assert list(tg.get_matrix().shape) == [N, 4, 4]


def test_sdf_serial_chain():
    print("Test building the serial chain from sdf file")
    chain = pk.build_serial_chain_from_sdf(open(os.path.join(UM_ARM_LAB_ASSETS, "simple_arm.sdf")).read(), 'arm_wrist_roll')
    chain = chain.to(dtype=torch.float64)
    tg = fk.forward_kinematics(chain,[0., math.pi / 2.0, -0.5, 0.])
    pos, rot = quat_pos_from_transform3d(tg)

    if quaternion_close(rot, torch.tensor([0.70710678, 0., 0., 0.70710678], dtype=torch.float64)):
        print("\tRotation Quaternion matches the expected value.")
    assert quaternion_close(rot, torch.tensor([0.70710678, 0., 0., 0.70710678], dtype=torch.float64))
    if torch.allclose(pos, torch.tensor([1.05, 0.55, 0.5], dtype=torch.float64)):
        print("\tPosition Quaternion matches the expected value.")
    assert torch.allclose(pos, torch.tensor([1.05, 0.55, 0.5], dtype=torch.float64))

def test_cuda():
    print("Test CUDA")
    if torch.cuda.is_available():
        d = "cuda"
        dtype = torch.float64
        chain = pk.build_chain_from_sdf(open(os.path.join(UM_ARM_LAB_ASSETS, "simple_arm.sdf")).read())
        # noinspection PyUnusedLocal
        chain = chain.to(dtype=dtype, device=d)

        # NOTE: do it twice because we previously had an issue with default arguments
        #  like joint=Joint() causing spooky behavior
        chain = pk.build_chain_from_sdf(open(os.path.join(UM_ARM_LAB_ASSETS, "simple_arm.sdf")).read())
        chain = chain.to(dtype=dtype, device=d)

        ret = fk.forward_kinematics(chain,{
            'arm_shoulder_pan_joint': 0,
            'arm_elbow_pan_joint': math.pi / 2.0,
            'arm_wrist_lift_joint': -0.5,
            'arm_wrist_roll_joint': 0,
        })
        tg = ret['arm_wrist_roll']
        pos, rot = quat_pos_from_transform3d(tg)
        if quaternion_close(rot, torch.tensor([0.70710678, 0., 0., 0.70710678], dtype=dtype, device=d)):
            print("\tRotation Quaternions front_left_foot match.")
        assert quaternion_close(rot, torch.tensor([0.70710678, 0., 0., 0.70710678], dtype=dtype, device=d))
        if torch.allclose(pos, torch.tensor([1.05, 0.55, 0.5], dtype=dtype, device=d)):
            print("\tPosition Quaternions front_left_foot match.")
        assert torch.allclose(pos, torch.tensor([1.05, 0.55, 0.5], dtype=dtype, device=d))

        
        

        data = '<robot name="test_robot">' \
               '<link name="link1" />' \
               '<link name="link2" />' \
               '<link name="link3" />' \
               '<joint name="joint1" type="revolute">' \
               '<origin xyz="1.0 0.0 0.0"/>' \
               '<parent link="link1"/>' \
               '<child link="link2"/>' \
               '</joint>' \
               '<joint name="joint2" type="revolute">' \
               '<origin xyz="1.0 0.0 0.0"/>' \
               '<parent link="link2"/>' \
               '<child link="link3"/>' \
               '</joint>' \
               '</robot>'
        chain = pk.build_serial_chain_from_urdf(data, 'link3')
        chain = chain.to(dtype=dtype, device=d)
        N = 20
        th_batch = torch.rand(N, 2).to(device=d, dtype=dtype)
        tg_batch = fk.forward_kinematics(chain,th_batch)
        m = tg_batch.get_matrix()
        for i in range(N):
            tg = fk.forward_kinematics(chain, th_batch[i])
            assert torch.allclose(tg.get_matrix().view(4, 4), m[i])
        print("\tDimensionalities of forward kinematics batch matches.")


# FIXME: comment out because compound joints are no longer implemented
# def test_fk_mjcf_humanoid():
#     chain = pk.build_chain_from_mjcf(open(os.path.join(TEST_DIR, "humanoid.xml")).read())
#     print(chain)
#     print(chain.get_joint_parameter_names())
#     th = {'left_knee': 0.0, 'right_knee': 0.0}
#     ret = fk.forward_kinematics(chain,th)
#     print(ret)


def test_mjcf_slide_joint_parsing():
    print("Test Parsing")
    # just testing that we can parse it without error
    # the slide joint is not actually of a link to another link, but instead of the base to the world
    # which we do not represent
    chain = pk.build_chain_from_mjcf(open(os.path.join(UM_ARM_LAB_ASSETS, "hopper.xml")).read())
    print("\t",chain.get_joint_parameter_names())
    print("\t",chain.get_frame_names())


def test_fk_val():
    print("Test Forward Kinematics Values")
    chain = pk.build_chain_from_mjcf(open(os.path.join(UM_ARM_LAB_ASSETS, "val.xml")).read())
    chain = chain.to(dtype=torch.float64)
    ret = fk.forward_kinematics(chain,torch.zeros([1000, chain.n_joints], dtype=torch.float64))
    tg = ret['drive45']
    pos, rot = quat_pos_from_transform3d(tg)
    torch.set_printoptions(precision=6, sci_mode=False)
    if quaternion_close(rot, torch.tensor([0.5, 0.5, -0.5, 0.5], dtype=torch.float64)):
        print("\tRotation Quaternion matches the expected value.")
    assert quaternion_close(rot, torch.tensor([0.5, 0.5, -0.5, 0.5], dtype=torch.float64))
    if torch.allclose(pos, torch.tensor([-0.225692, 0.259045, 0.262139], dtype=torch.float64)):
        print("\tPosition Quaternion matches the expected value.")
    assert torch.allclose(pos, torch.tensor([-0.225692, 0.259045, 0.262139], dtype=torch.float64))


def test_fk_partial_batched_dict():
    print("Test subsetting joints with a dictionary")
    # Test that you can pass in dict of batched joint configs for a subset of the joints
    chain = pk.build_serial_chain_from_mjcf(open(os.path.join(UM_ARM_LAB_ASSETS, "val.xml")).read(), 'left_tool')
    th = {
        'joint56': torch.zeros([1000], dtype=torch.float64),
        'joint57': torch.zeros([1000], dtype=torch.float64),
        'joint41': torch.zeros([1000], dtype=torch.float64),
        'joint42': torch.zeros([1000], dtype=torch.float64),
        'joint43': torch.zeros([1000], dtype=torch.float64),
        'joint44': torch.zeros([1000], dtype=torch.float64),
        'joint45': torch.zeros([1000], dtype=torch.float64),
        'joint46': torch.zeros([1000], dtype=torch.float64),
        'joint47': torch.zeros([1000], dtype=torch.float64),
    }
    torch.set_printoptions(precision=2, sci_mode=False)
    chain = chain.to(dtype=torch.float64)
    tg = fk.forward_kinematics(chain,th)
    print("\tRotation/Position Transform Vector:")
    print("\t",tg)
    print("\n")

def test_fk_partial_batched():
    # Test that you can pass in dict of batched joint configs for a subset of the joints
    print("Test subsetting joints")
    chain = pk.build_serial_chain_from_mjcf(open(os.path.join(UM_ARM_LAB_ASSETS, "val.xml")).read(), 'left_tool')
    th = torch.zeros([1000, 9], dtype=torch.float64)
    torch.set_printoptions(precision=2, sci_mode=False)
    chain = chain.to(dtype=torch.float64)
    tg = fk.forward_kinematics(chain,th)
    print("\tRotation/Position Transform Vector:")
    print("\t",tg)
    print("\n")


def test_ur5_fk():
    print("Test ur5 forward kinematics.")
    urdf = os.path.join(UM_ARM_LAB_ASSETS, "ur5.urdf")
    pk_chain = pk.build_serial_chain_from_urdf(open(urdf).read(), 'ee_link', 'base_link')
    th = [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0]

    try:
        import ikpy.chain
        ik_chain = ikpy.chain.Chain.from_urdf_file(urdf,
                                                   active_links_mask=[False, True, True, True, True, True, True, False])
        ik_ret = fk.forward_kinematics(ik_chain,[0, *th, 0])
    except ImportError:
        ik_ret = [[-6.44330720e-18, 3.58979314e-09, -1.00000000e+00, 5.10955359e-01],
                  [1.00000000e+00, 1.79489651e-09, 0.00000000e+00, 1.91450000e-01],
                  [1.79489651e-09, -1.00000000e+00, -3.58979312e-09, 6.00114361e-01],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        print("\tImport Error")

    ret = fk.forward_kinematics(pk_chain,th, end_only=True)
    print("\t",ret.get_matrix())
    ik_ret = torch.tensor(ik_ret, dtype=ret.dtype)
    print("\t",ik_ret)
    if torch.allclose(ik_ret, ret.get_matrix(), atol=1e-6):
        print("\tur5 Forward Kinematics functional.")
    assert torch.allclose(ik_ret, ret.get_matrix(), atol=1e-6)


if __name__ == "__main__":
    test_fk_partial_batched()
    test_fk_partial_batched_dict()
    test_fk_val()
    test_sdf_serial_chain()
    test_urdf_serial()
    test_fkik()
    test_fk_simple_arm()
    test_fk_mjcf()
    test_cuda()
    test_urdf()
    # test_fk_mjcf_humanoid()
    test_mjcf_slide_joint_parsing()
    test_ur5_fk()
    print("All tests passed.")
