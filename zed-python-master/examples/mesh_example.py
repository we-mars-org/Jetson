########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Mesh sample shows mesh information after filtering and applying texture on frames. The mesh and its filter
    parameters can be saved.
"""
import sys
import pyzed.camera as zcam
import pyzed.core as core
import pyzed.mesh as mesh
import pyzed.types as tp


def main():

    if len(sys.argv) != 2:
        print("Please specify path to .svo file.")
        exit()

    filepath = sys.argv[1]
    print("Reading SVO file: {0}".format(filepath))

    cam = zcam.PyZEDCamera()
    init = zcam.PyInitParameters(svo_input_filename=filepath)
    status = cam.open(init)
    if status != tp.PyERROR_CODE.PySUCCESS:
        print(repr(status))
        exit()

    runtime = zcam.PyRuntimeParameters()
    spatial = zcam.PySpatialMappingParameters()
    transform = core.PyTransform()
    tracking = zcam.PyTrackingParameters(transform)

    cam.enable_tracking(tracking)
    cam.enable_spatial_mapping(spatial)

    pymesh = mesh.PyMesh()
    print("Processing...")
    for i in range(200):
        cam.grab(runtime)
        cam.request_mesh_async()

    cam.extract_whole_mesh(pymesh)
    cam.disable_tracking()
    cam.disable_spatial_mapping()

    filter_params = mesh.PyMeshFilterParameters()
    filter_params.set(mesh.PyFILTER.PyFILTER_HIGH)
    print("Filtering params : {0}.".format(pymesh.filter(filter_params)))

    apply_texture = pymesh.apply_texture(mesh.PyMESH_TEXTURE_FORMAT.PyMESH_TEXTURE_RGBA)
    print("Applying texture : {0}.".format(apply_texture))
    print_mesh_information(pymesh, apply_texture)

    save_filter(filter_params)
    save_mesh(pymesh)
    cam.close()
    print("\nFINISH")


def print_mesh_information(pymesh, apply_texture):
    while True:
        res = input("Do you want to display mesh information? [y/n]: ")
        if res == "y":
            if apply_texture:
                print("Vertices : \n{0} \n".format(pymesh.vertices))
                print("Uv : \n{0} \n".format(pymesh.uv))
                print("Normals : \n{0} \n".format(pymesh.normals))
                print("Triangles : \n{0} \n".format(pymesh.triangles))
                break
            else:
                print("Cannot display information of the mesh.")
                break
        if res == "n":
            print("Mesh information will not be displayed.")
            break
        else:
            print("Error, please enter [y/n].")


def save_filter(filter_params):
    while True:
        res = input("Do you want to save the mesh filter parameters? [y/n]: ")
        if res == "y":
            params = tp.PyERROR_CODE.PyERROR_CODE_FAILURE
            while params != tp.PyERROR_CODE.PySUCCESS:
                filepath = input("Enter filepath name : ")
                params = filter_params.save(filepath)
                print("Saving mesh filter parameters: {0}".format(repr(params)))
                if params:
                    break
                else:
                    print("Help : you must enter the filepath + filename without extension.")
            break
        elif res == "n":
            print("Mesh filter parameters will not be saved.")
            break
        else:
            print("Error, please enter [y/n].")


def save_mesh(pymesh):
    while True:
        res = input("Do you want to save the mesh? [y/n]: ")
        if res == "y":
            msh = tp.PyERROR_CODE.PyERROR_CODE_FAILURE
            while msh != tp.PyERROR_CODE.PySUCCESS:
                filepath = input("Enter filepath name: ")
                msh = pymesh.save(filepath)
                print("Saving mesh: {0}".format(repr(msh)))
                if msh:
                    break
                else:
                    print("Help : you must enter the filepath + filename without extension.")
            break
        elif res == "n":
            print("Mesh will not be saved.")
            break
        else:
            print("Error, please enter [y/n].")

if __name__ == "__main__":
    main()
