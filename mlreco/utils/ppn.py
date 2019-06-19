from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def contains(meta, point, point_type="3d"):
    if point_type == '3d':
        return point.x() >= meta.min_x() and point.y() >= meta.min_y() \
            and point.z() >= meta.min_z() and point.x() <= meta.max_x() \
            and point.y() <= meta.max_y() and point.z() <= meta.max_z()
    else:
        return point.x() >= meta.min_x() and point.x() <= meta.max_x() \
            and point.y() >= meta.min_y() and point.y() <= meta.max_y()


def get_ppn_info(particle_v, meta, point_type="3d", min_voxel_count=5, min_energy_deposit=0.05):
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    # from larcv import larcv
    gt_positions = []
    for particle in particle_v:
        pdg_code = particle.pdg_code()
        prc = particle.creation_process()
        # Skip particle under some conditions
        if (particle.energy_deposit() < min_energy_deposit or particle.num_voxels() < min_voxel_count):
            continue
        if pdg_code > 1000000000:  # skipping nucleus trackid
            continue
        if pdg_code == 11 or pdg_code == 22 or pdg_code == -11:  # Shower
            if not contains(meta, particle.first_step(), point_type=point_type):
                continue
            # # Skipping delta ray
            # if (particle.parent_pdg_code() == 13 and prc == "muIoni") or prc == "hIoni":
            #     continue

        # Determine point type
        if (pdg_code == 2212):
            gt_type = 0
        elif pdg_code != 22 and pdg_code != 11:
            gt_type = 1
        elif pdg_code == 22:
            gt_type = 2
        else:
            if prc == "primary" or prc == "nCapture" or prc == "conv":
                gt_type = 2
            elif prc == "muIoni" or prc == "hIoni":
                gt_type = 3
            elif prc == "muMinusCaptureAtRest" or prc == "muPlusCaptureAtRest" or prc == "Decay":
                gt_type = 4

        # TODO deal with different 2d projections
        # Register start point
        x = particle.first_step().x()
        y = particle.first_step().y()
        z = particle.first_step().z()
        if point_type == '3d':
            x = (x - meta.min_x()) / meta.size_voxel_x()
            y = (y - meta.min_y()) / meta.size_voxel_y()
            z = (z - meta.min_z()) / meta.size_voxel_z()
            gt_positions.append([x, y, z, gt_type])
        else:
            x = (x - meta.min_x()) / meta.pixel_width()
            y = (y - meta.min_y()) / meta.pixel_height()
            gt_positions.append([x, y, gt_type])

        # Register end point (for tracks only)
        if gt_type == 0 or gt_type == 1:
            x = particle.last_step().x()
            y = particle.last_step().y()
            z = particle.last_step().z()
            if point_type == '3d':
                x = (x - meta.min_x()) / meta.size_voxel_x()
                y = (y - meta.min_y()) / meta.size_voxel_y()
                z = (z - meta.min_z()) / meta.size_voxel_z()
                gt_positions.append([x, y, z, gt_type])
            else:
                x = (x - meta.min_x()) / meta.pixel_width()
                y = (y - meta.min_y()) / meta.pixel_height()
                gt_positions.append([x, y, gt_type])

    return np.array(gt_positions)
