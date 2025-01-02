import numpy as np
import torch
import librosa
import laion_clap
from torchsynth.synth import Voice
from torchsynth.config import SynthConfig
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Union, Dict, Tuple, List
import multiprocessing
from pymoo.core.problem import StarmapParallelization
import json

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

@dataclass
class SynthParameter:
    name: Tuple[str, str]
    min_val: float
    max_val: float

class SynthProblem(Problem):
    def __init__(self, synth_optimizer, target_embedding, **kwargs):
        self.synth_optimizer = synth_optimizer
        self.target_embedding = target_embedding
        
        self.param_specs = [
            # Keyboard parameters
            SynthParameter(("keyboard", "midi_f0"), 30.0, 90.0),
            SynthParameter(("keyboard", "duration"), 0.5, 4.0),
            
            # ADSR 1 parameters
            SynthParameter(("adsr_1", "attack"), 0.001, 2.0),  # Increased max
            SynthParameter(("adsr_1", "decay"), 0.001, 2.0),   # Increased max
            SynthParameter(("adsr_1", "sustain"), 0.0, 1.0),
            SynthParameter(("adsr_1", "release"), 0.001, 3.0), # Increased max
            SynthParameter(("adsr_1", "alpha"), 1.0, 6.0),
            
            # ADSR 2 parameters
            SynthParameter(("adsr_2", "attack"), 0.001, 2.0),
            SynthParameter(("adsr_2", "decay"), 0.001, 2.0),
            SynthParameter(("adsr_2", "sustain"), 0.0, 1.0),
            SynthParameter(("adsr_2", "release"), 0.001, 3.0),
            SynthParameter(("adsr_2", "alpha"), 1.0, 6.0),
            
            # LFO 1 parameters
            SynthParameter(("lfo_1", "frequency"), 0.1, 20.0),  # Increased max
            SynthParameter(("lfo_1", "mod_depth"), 0.0, 1.0),
            SynthParameter(("lfo_1", "initial_phase"), 0.0, 3.1416),
            SynthParameter(("lfo_1", "sin"), 0.0, 1.0),
            SynthParameter(("lfo_1", "tri"), 0.0, 1.0),
            SynthParameter(("lfo_1", "saw"), 0.0, 1.0),
            SynthParameter(("lfo_1", "rsaw"), 0.0, 1.0),
            SynthParameter(("lfo_1", "sqr"), 0.0, 1.0),
            
            # LFO 2 parameters
            SynthParameter(("lfo_2", "frequency"), 0.1, 20.0),
            SynthParameter(("lfo_2", "mod_depth"), 0.0, 1.0),
            SynthParameter(("lfo_2", "initial_phase"), 0.0, 3.1416),
            SynthParameter(("lfo_2", "sin"), 0.0, 1.0),
            SynthParameter(("lfo_2", "tri"), 0.0, 1.0),
            SynthParameter(("lfo_2", "saw"), 0.0, 1.0),
            SynthParameter(("lfo_2", "rsaw"), 0.0, 1.0),
            SynthParameter(("lfo_2", "sqr"), 0.0, 1.0),
            
            # LFO 1 Amplitude ADSR
            SynthParameter(("lfo_1_amp_adsr", "attack"), 0.001, 2.0),
            SynthParameter(("lfo_1_amp_adsr", "decay"), 0.001, 2.0),
            SynthParameter(("lfo_1_amp_adsr", "sustain"), 0.0, 1.0),
            SynthParameter(("lfo_1_amp_adsr", "release"), 0.001, 3.0),
            SynthParameter(("lfo_1_amp_adsr", "alpha"), 1.0, 6.0),
            
            # LFO 2 Amplitude ADSR
            SynthParameter(("lfo_2_amp_adsr", "attack"), 0.001, 2.0),
            SynthParameter(("lfo_2_amp_adsr", "decay"), 0.001, 2.0),
            SynthParameter(("lfo_2_amp_adsr", "sustain"), 0.0, 1.0),
            SynthParameter(("lfo_2_amp_adsr", "release"), 0.001, 3.0),
            SynthParameter(("lfo_2_amp_adsr", "alpha"), 1.0, 6.0),
            
            # LFO 1 Rate ADSR
            SynthParameter(("lfo_1_rate_adsr", "attack"), 0.001, 2.0),
            SynthParameter(("lfo_1_rate_adsr", "decay"), 0.001, 2.0),
            SynthParameter(("lfo_1_rate_adsr", "sustain"), 0.0, 1.0),
            SynthParameter(("lfo_1_rate_adsr", "release"), 0.001, 3.0),
            SynthParameter(("lfo_1_rate_adsr", "alpha"), 1.0, 6.0),
            
            # LFO 2 Rate ADSR
            SynthParameter(("lfo_2_rate_adsr", "attack"), 0.001, 2.0),
            SynthParameter(("lfo_2_rate_adsr", "decay"), 0.001, 2.0),
            SynthParameter(("lfo_2_rate_adsr", "sustain"), 0.0, 1.0),
            SynthParameter(("lfo_2_rate_adsr", "release"), 0.001, 3.0),
            SynthParameter(("lfo_2_rate_adsr", "alpha"), 1.0, 6.0),
            
            # Modulation Matrix
            SynthParameter(("mod_matrix", "adsr_1->vco_1_pitch"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "adsr_1->vco_1_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "adsr_1->vco_2_pitch"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "adsr_1->vco_2_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "adsr_1->noise_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "adsr_2->vco_1_pitch"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "adsr_2->vco_1_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "adsr_2->vco_2_pitch"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "adsr_2->vco_2_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "adsr_2->noise_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_1->vco_1_pitch"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_1->vco_1_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_1->vco_2_pitch"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_1->vco_2_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_1->noise_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_2->vco_1_pitch"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_2->vco_1_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_2->vco_2_pitch"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_2->vco_2_amp"), 0.0, 1.0),
            SynthParameter(("mod_matrix", "lfo_2->noise_amp"), 0.0, 1.0),
            
            # VCO 1 parameters
            SynthParameter(("vco_1", "tuning"), -24.0, 24.0),  # Wider range
            SynthParameter(("vco_1", "mod_depth"), -24.0, 24.0),  # Wider range
            SynthParameter(("vco_1", "initial_phase"), 0.0, 3.1416),
            
            # VCO 2 parameters
            SynthParameter(("vco_2", "tuning"), -24.0, 24.0),  # Wider range
            SynthParameter(("vco_2", "mod_depth"), -24.0, 24.0),  # Wider range
            SynthParameter(("vco_2", "initial_phase"), 0.0, 3.1416),
            SynthParameter(("vco_2", "shape"), 0.0, 1.0),
            
            # Mixer parameters
            SynthParameter(("mixer", "vco_1"), 0.0, 1.0),
            SynthParameter(("mixer", "vco_2"), 0.0, 1.0),
            SynthParameter(("mixer", "noise"), 0.0, 1.0)
        ]
        
        super().__init__(
            n_var=len(self.param_specs),
            n_obj=1,
            xl=np.array([p.min_val for p in self.param_specs], dtype=np.float32),
            xu=np.array([p.max_val for p in self.param_specs], dtype=np.float32),
            **kwargs
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate entire population at once"""
        n_solutions = len(X)
        
        # Update synth config for batch processing
        self.synth_optimizer.update_batch_size(n_solutions)
        
        # Create batched parameter dictionary
        param_dict = {}
        for j, param in enumerate(self.param_specs):
            values = torch.tensor(X[:, j], device=self.synth_optimizer.device, dtype=torch.float32)
            param_dict[param.name] = values
        
        try:
            # Generate all audio in one batch
            self.synth_optimizer.voice.set_parameters(param_dict)
            audio_batch, _, _ = self.synth_optimizer.voice()
            
            # Process all audio for CLAP - detach before converting to numpy
            audio_batch = audio_batch.detach().cpu().numpy()
            audio_batch = torch.from_numpy(
                int16_to_float32(float32_to_int16(audio_batch))
            ).float()
            
            # Get embeddings for entire batch
            embeddings = self.synth_optimizer.clap.get_audio_embedding_from_data(
                x=audio_batch,
                use_tensor=True
            )
            
            # Calculate losses for entire batch
            losses = 1 - torch.sum(embeddings * self.target_embedding, dim=1)
            # Detach before converting to numpy
            out["F"] = losses.detach().cpu().numpy()
            
        except Exception as e:
            print(f"Error in batch evaluation: {str(e)}")
            out["F"] = np.full(n_solutions, float('inf'))

class SynthOptimizer:
    def __init__(self, sample_rate=48000, buffer_size_seconds=4.0, initial_batch_size=1):
        # Initialize CLAP model
        self.clap = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap.load_ckpt()
        
        self.sample_rate = sample_rate
        self.buffer_size_seconds = buffer_size_seconds
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize with initial batch size
        self.update_batch_size(initial_batch_size)
    
    def update_batch_size(self, batch_size):
        """Update synthesizer with new batch size"""
        self.config = SynthConfig(
            batch_size=batch_size,
            sample_rate=self.sample_rate,
            buffer_size_seconds=self.buffer_size_seconds,
            reproducible=False
        )
        self.voice = Voice(synthconfig=self.config).to(self.device)
        
        # Ensure all parameters are float32
        for param in self.voice.parameters():
            param.data = param.data.float()
    
    def get_audio_embedding(self, audio_path: str) -> torch.Tensor:
        """Get embedding for target audio file"""
        audio_data, _ = librosa.load(audio_path, sr=48000)
        audio_data = audio_data.reshape(1, -1)
        audio_tensor = torch.from_numpy(
            int16_to_float32(float32_to_int16(audio_data))
        ).float()
        
        return self.clap.get_audio_embedding_from_data(
            x=audio_tensor,
            use_tensor=True
        )
    
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text description"""
        return self.clap.get_text_embedding([text], use_tensor=True)
    
    def save_parameters(self, parameters: Dict, filepath: str):
        """Save parameters to JSON file"""
        # Convert parameters to serializable format
        save_dict = {}
        for (module, param), value in parameters.items():
            # Convert tensor to float
            save_dict[f"{module}/{param}"] = float(value.cpu().numpy()[0])
            
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=4)
    
    def load_parameters(self, filepath: str) -> Dict:
        """Load parameters from JSON file"""
        with open(filepath, 'r') as f:
            save_dict = json.load(f)
            
        # Convert back to proper parameter format
        parameters = {}
        for key, value in save_dict.items():
            module, param = key.split('/')
            parameters[(module, param)] = torch.tensor([value], 
                                                     device=self.device,
                                                     dtype=torch.float32)
            
        return parameters
    
    def optimize(self, target: Union[str, torch.Tensor], 
                population_size: int = 50,
                n_generations: int = 100,
                n_processes: int = 4,
                is_text: bool = False,
                save_path: Optional[str] = None) -> Tuple[Dict, float]:
        """
        Optimize synth parameters to match target using parallel processing
        """
        if isinstance(target, str):
            if is_text:
                target_embedding = self.get_text_embedding(target)
            else:
                target_embedding = self.get_audio_embedding(target)
        else:
            target_embedding = target
        
        # Setup parallel processing
        pool = multiprocessing.Pool(n_processes)
        runner = StarmapParallelization(pool.starmap)
        
        # Setup optimization problem with parallelization
        problem = SynthProblem(
            self, 
            target_embedding,
            elementwise_runner=runner
        )
        
        # Configure differential evolution
        algorithm = DE(
            pop_size=population_size,
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.3,
            dither="vector",
            jitter=False,
        )
        
        # Run optimization
        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_generations),
            seed=1,
            verbose=True
        )
        
        pool.close()
        
        # Convert best solution to parameter dictionary
        best_params = {}
        for i, param in enumerate(problem.param_specs):
            best_params[param.name] = torch.tensor([float(res.X[i])], 
                                                 device=self.device,
                                                 dtype=torch.float32)
            
        # Save parameters if path provided
        if save_path:
            self.save_parameters(best_params, save_path)
            
        return best_params, float(res.F[0])
    
    def synthesize_and_save(self, parameters: Dict, output_path: str):
        """Generate audio using given parameters and save to file"""
        # Ensure we're in single-sample mode
        self.update_batch_size(1)
        self.voice.set_parameters(parameters)
        audio, _, _ = self.voice()
        import torchaudio
        torchaudio.save(output_path, audio.cpu(), self.config.sample_rate)

# Example usage
if __name__ == "__main__":
    print("Initializing synthesizer...")
    optimizer = SynthOptimizer(initial_batch_size=1)
    
    # Example with text target
    text_prompt = "chirping birds"
    print(f"Optimizing to match text: {text_prompt}")
    
    best_params, final_loss = optimizer.optimize(
        text_prompt,
        population_size=30,
        n_generations=30,
        n_processes=4,
        is_text=True,
        save_path="chirpingbirds.json"  # Parameters will be saved here
    )
    
    optimizer.synthesize_and_save(best_params, "chirpingbirds.wav")
    print(f"Text matching complete. Final loss: {final_loss:.4f}")
    print("Parameters found:", best_params)
    print("Parameters saved to: chirpingbirds.json")