# clustering.py
import os
import csv
import json
import numpy as np
import freud
import hoomd

class ClusterValidator(hoomd.custom.Action):
    """
    Custom HOOMD action to compute cluster masses and velocity correlations.
    """
    def __init__(self, filename="./Plot_data/cluster_validation.csv", r_max=5.0, dr=0.1):
        super().__init__()
        self.filename = filename
        self.r_max = r_max
        self.dr = dr
        self._header_written = False

    def _compute_velocity_correlation(self, box, positions, velocities):
        """Internal helper method to calculate spatial velocity correlation."""
        max_allowed_r = np.min(box.L) / 2.0 
        safe_r_max = max_allowed_r - 1e-4
        actual_r_max = min(self.r_max, safe_r_max)
        
        if self.r_max > safe_r_max:
            print(f"[Warning] Requested r_max={self.r_max} is too large for box dimensions {box.L}. Capping to {actual_r_max:.4f}")
        # ------------------------

   
        aq = freud.locality.AABBQuery(box, positions)
        query_result = aq.query(positions, dict(r_max=actual_r_max, exclude_ii=True))
        nlist = query_result.toNeighborList()
        
   
        distances = nlist.distances
        i_indices = nlist.query_point_indices
        j_indices = nlist.point_indices
        

        v_i = velocities[i_indices]
        v_j = velocities[j_indices]
        dot_products = np.sum(v_i * v_j, axis=1)

        bins = np.arange(0, actual_r_max + self.dr, self.dr)
        bin_centers = bins[:-1] + self.dr / 2.0
        

        sum_correlations, _ = np.histogram(distances, bins=bins, weights=dot_products)
        pair_counts, _ = np.histogram(distances, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            C_vv = sum_correlations / pair_counts
            C_vv[pair_counts == 0] = 0.0  # Set empty bins to 0
            
        return bin_centers, C_vv

    def act(self, timestep):
        snap = self._state.get_snapshot()
        if snap.communicator.rank == 0:
            positions = snap.particles.position
            velocities = snap.particles.velocity  
            box_array = snap.configuration.box
            
            box = freud.box.Box.from_box(box_array)
            
            # --- 1. Compute Cluster Properties ---
            cl = freud.cluster.Cluster()
            cl.compute(system=(box, positions), neighbors={'r_max': 1.2})
            
            cl_props = freud.cluster.ClusterProperties()
            cl_props.compute(system=(box, positions), cluster_idx=cl.cluster_idx)
            
            cluster_sizes = cl_props.sizes
            
            mass_dict = {}
            for cluster_id, size in enumerate(cluster_sizes):
                mass_dict[cluster_id] = size / ((2**(1/6))**3)
            
            masses = np.array(list(mass_dict.values()))
            average_mass = np.mean(masses) if len(masses) > 0 else 0
            
            # --- 2. Compute Velocity Correlation ---
            bin_centers, C_vv = self._compute_velocity_correlation(box, positions, velocities)
            
            # --- 3. Terminal Diagnostics ---
            print(f"\n--- Timestep {timestep} Validation ---")
            print(f"Loaded {len(positions)} particles.")
            print(f"Found {cl.num_clusters} clusters.")
            print(f"Average Mass: {average_mass:.4f}")
            print("-" * 30)

           
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            file_exists = os.path.isfile(self.filename)
            
            # --- 4. Populate CSV Dataset ---
            with open(self.filename, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                if not file_exists and not self._header_written:
                    writer.writerow([
                        'timestep', 'num_particles', 'num_clusters', 
                        'average_mass', 'cluster_mass_dict', 
                        'c_vv_bins', 'c_vv_values'
                    ])
                    self._header_written = True
                
              
                dict_string = json.dumps(mass_dict)
                bins_string = json.dumps(bin_centers.tolist())
                cvv_string = json.dumps(C_vv.tolist())
                
                writer.writerow([
                    timestep, len(positions), cl.num_clusters, 
                    average_mass, dict_string, bins_string, cvv_string
                ])
            
            print(f"Timestep {timestep}: Logged data to {self.filename}")