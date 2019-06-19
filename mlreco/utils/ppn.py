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

def pass_particle(gt_type, start, end, energy_deposit, vox_count):

    if (np.power((start.x()-end.x()),2) + np.power((start.y()-end.y()),2) + np.power((start.z()-end.z()),2)) < 6.25:
        return True
    if gt_type == 0: return vox_count<7 or energy_deposit < 50.
    if gt_type == 1: return vox_count<7 or energy_deposit < 10.
    if gt_type == 2: return vox_count<7 or energy_deposit < 1.
    if gt_type == 3: return vox_count<5 or energy_deposit < 5.
    if gt_type == 4: return vox_count<5 or energy_deposit < 5.

def get_ppn_info(particle_v, meta, point_type="3d", min_voxel_count=7, min_energy_deposit=10, annotate=True):
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    # from larcv import larcv
    gt_positions = []
    annotations = []
    for particle in particle_v:
        pdg_code = abs(particle.pdg_code())
        prc = particle.creation_process()
        # Skip particle under some conditions
        if particle.energy_deposit() < min_energy_deposit or particle.num_voxels() < min_voxel_count:
            continue
        if pdg_code > 1000000000:  # skipping nucleus trackid
            continue
        if pdg_code == 11 or pdg_code == 22:  # Shower
            if not contains(meta, particle.first_step(), point_type=point_type):
                continue
            # Skipping delta ray
            #if particle.parent_pdg_code() == 13 and particle.creation_process() == "muIoni":
            #    continue

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

        #if pass_particle(gt_type,particle.first_step(),particle.last_step(),particle.energy_deposit(),particle.num_voxels()):
        #    continue
                         
        annotation=''
        if annotate:
            annotation='PDG %d Track %d Parent %d E %.2f Dep %.2f Npx %d' % (particle.pdg_code(),particle.track_id(),particle.parent_track_id(),particle.energy_init(),particle.energy_deposit(),particle.num_voxels())
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
            if annotate:
                annotations.append(annotation + ' start')
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
                if annotate:
                    annotations.append(annotation + ' end')
            else:
                x = (x - meta.min_x()) / meta.pixel_width()
                y = (y - meta.min_y()) / meta.pixel_height()
                gt_positions.append([x, y, gt_type])
    
    return np.array(gt_positions), np.array(annotations)
    #return np.array(gt_positions)
