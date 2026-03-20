from run_pretrained_openfold import main
from openfold.model.heads import DistogramHead
from argparse import Namespace
import os
import numpy as np
import torch

class Argsafm:

    def __init__(self):

        self.args = Namespace()

        # Default parameters
        self.args.use_single_seq_mode = False
        self.args.jax_param_path = None
        self.args.save_outputs = False
        self.args.preset = 'full_dbs'
        self.args.output_postfix = None
        self.args.data_random_seed = None
        self.args.trace_model = False
        self.args.subtract_plddt = False
        self.args.long_sequence_inference = False
        self.args.multimer_ri_gap = 0
        self.args.max_template_date='2021-12-19'
        self.args.obsolete_pdbs_path=None
        self.args.release_dates_path=None
        self.args.cif_output=False

        # Database parameters
        self.args.template_mmcif_dir = 'pdb_mmcif/mmcif_files/'
        self.args.uniref90_database_path = 'uniref90/uniref90.fasta'
        self.args.mgnify_database_path = 'mgnify/mgy_clusters.fa'
        self.args.pdb70_database_path = 'pdb70/pdb70'
        self.args.pdb_seqres_database_path = 'pdb_seqres/pdb_seqres.txt'
        self.args.uniref30_database_path = 'uniclust30_2018_08/uniclust30_2018_08'
        self.args.uniclust30_database_path = 'uniclust30_2018_08/uniclust30_2018_08'
        self.args.uniprot_database_path = 'uniprot/uniprot.fasta'
        self.args.bfd_database_path = 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'

        # Command parameters
        self.args.jackhmmer_binary_path = 'jackhmmer'
        self.args.hhblits_binary_path = 'hhblits'
        self.args.hhsearch_binary_path = 'hhsearch'
        self.args.hmmsearch_binary_path = 'hmmsearch'
        self.args.hmmbuild_binary_path = 'hmmbuild'
        self.args.kalign_binary_path = 'kalign'

        # Model parameters
        self.args.config_preset ="model_1_multimer_v3"
        self.args.jax_param_path = 'openfold/resources/params/params_model_3_multimer_v3.npz'
        self.args.openfold_checkpoint_path = None #
        self.args.cpus = 32
        self.args.skip_relaxation = True
        self.args.use_precomputed_alignments = None
        self.args.save_outputs = False
        self.args.use_deepspeed_evoformer_attention=False
        self.args.experiment_config_json = None
        self.args.use_cuequivariance_attention=False
        self.args.use_cuequivariance_multiplicative_update=False


    def get_args(self):
        return self.args
    def set_fasta_path(self, value):
        self.args.fasta_dir = value
    def set_out_path(self, value):
        self.args.output_dir = value
    def set_device(self, value):
        self.args.model_device = value
    def set_alignments(self, value):
        self.args.use_precomputed_alignments = value
    def set_cpus(self, value):
        self.args.cpus = value


#### Run AFM
device = "cuda:1"

run_path = "example/8G4K" 
model_name = "templates"
parseafm =  Argsafm()
parseafm.set_fasta_path(f"{run_path}/{model_name}/fasta")
parseafm.set_out_path(f"{run_path}/{model_name}/results")
parseafm.set_alignments(f"{run_path}/{model_name}/alignments")
parseafm.set_device(device)
argsafm = parseafm.get_args()
main(argsafm)


#### Transform pair representations to contact probability maps
os.makedirs(f"{run_path}/{model_name}/results/distogram_1", exist_ok=True)
zlist=torch.from_numpy(np.load(f"{run_path}/{model_name}/results/H-L-A_1_zstack.npy"))

params = np.load('openfold/openfold/resources/params/params_model_3_multimer_v3.npz')
weight_key = 'alphafold/alphafold_iteration/distogram_head/half_logits//weights'
bias_key = 'alphafold/alphafold_iteration/distogram_head/half_logits//bias'

distogram_head = DistogramHead(c_z=128, no_bins=64)
distogram_head.linear.weight.data = torch.from_numpy(params[weight_key]).t()
distogram_head.linear.bias.data = torch.from_numpy(params[bias_key])
distogram_head.eval()

for i, z in enumerate(zlist):
    with torch.no_grad():
        if len(z.shape) == 3:
            z = z.unsqueeze(0)
        logits = distogram_head(z)
        probs = torch.softmax(logits, dim=-1)
        distance = probs.squeeze(0).cpu().numpy()
        contact_map = torch.sum(torch.from_numpy(distance)[:, :, :32], dim=2)
        filename = f"{run_path}/{model_name}/results/distogram_1/layer_{i+1}_contact_map.txt"
        np.savetxt(filename, np.array(contact_map))