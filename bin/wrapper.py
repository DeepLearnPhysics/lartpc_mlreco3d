# Import first here to avoid ROOT mixup
import os, sys, yaml, glob
current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
from analysis.run import main

assert len(sys.argv) >= 4
io_cfg = sys.argv[1]
analysis_cfg = sys.argv[2]
file_count_per_task = int(sys.argv[3])
sample = sys.argv[4]
job_name = sys.argv[5]
voxel_count_threshold = int(sys.argv[6])

io_cfg = yaml.load(open(io_cfg, 'r'), Loader=yaml.Loader)
analysis_cfg = yaml.load(open(analysis_cfg, 'r'), Loader=yaml.Loader)

# Set input files based on env variable
task_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)
if task_id is None:
    raise Exception("Environment variable SLRUM_ARRAY_TASK_ID is undefined.")
task_id = int(task_id)

#input_files = io_cfg['iotool']['dataset']['data_keys']
sample_dir = os.path.join("/sdf/group/neutrino/ldomine", sample)
input_files = [os.path.join(sample_dir, "larcv*.root")]
file_list = []
for f in input_files:
    file_list.extend(glob.glob(f))
file_list.sort()
file_list = file_list[(task_id-1) * file_count_per_task:task_id * file_count_per_task]
io_cfg['iotool']['dataset']['data_keys'] = file_list
io_cfg['iotool']['dataset']['limit_num_files'] = file_count_per_task+1

# MAke output directory
#out_dir = analysis_cfg['analysis']['log_dir']
sample_name = sample.split('/')[-1]
out_dir = os.path.join("/sdf/group/neutrino/ldomine/analysis", "log_%s_%s" % (sample_name, job_name))
job_id = os.environ.get('SLURM_JOB_ID', None)
if job_id is None:
    raise Exception("Environment variable SLURM_JOB_ID is undefined.")
job_id = int(job_id)
job_out_dir = os.path.join(out_dir, '%d_%d' % (job_id, task_id))
if not os.path.isdir(job_out_dir):
    #os.mkdir(job_out_dir)
    os.makedirs(job_out_dir)

# Make skip entries list based on voxel count
print("Scanning input files to exclude entries with voxel count > %d..." % voxel_count_threshold)
from ROOT import TChain
chainE = TChain("sparse3d_reco_cryoE_tree")
chainW = TChain("sparse3d_reco_cryoW_tree")
for f in file_list:
    chainE.AddFile(f)
    chainW.AddFile(f)
s = []
for e in range(chainE.GetEntries()):
    if e % 100 == 0:
        print("%d / %d" % (e, chainE.GetEntries()))
    chainE.GetEntry(e)
    chainW.GetEntry(e)
    recoE = getattr(chainE, "sparse3d_reco_cryoE_branch")
    recoW = getattr(chainW, "sparse3d_reco_cryoW_branch")
    nE = len(recoE.as_vector())
    nW = len(recoW.as_vector())
    if nE + nW > voxel_count_threshold:
        s.append(str(e))

with open(os.path.join(job_out_dir, "skip_entries.txt"), 'w') as f:
    f.write(",".join(s))
io_cfg['iotool']['dataset']['skip_event_list'] = os.path.join(job_out_dir, "skip_entries.txt")
print("... done. Skipping %d entries." % len(s))
sys.stdout.flush()

# Write configs to output dir
analysis_cfg['analysis']['log_dir'] = job_out_dir
with open(os.path.join(job_out_dir, 'fullchain.cfg'), 'w') as f:
    f.write(yaml.dump(io_cfg, default_flow_style=None))
with open(os.path.join(job_out_dir, 'analysis.cfg'), 'w') as f:
    f.write(yaml.dump(analysis_cfg, default_flow_style=None))

# Time to run analysis
main(os.path.join(job_out_dir, 'analysis.cfg'), os.path.join(job_out_dir, 'fullchain.cfg'))
