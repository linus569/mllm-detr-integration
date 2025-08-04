from pathlib import Path
import sys
import torch
import logging
from types import SimpleNamespace

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.model import VisionLanguageModel
from utils.config import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_state_dict():
    """Test saving and loading state dictionaries."""
    logger.info("Initializing test configuration")
    config = ExperimentConfig()
    config.detr_loss = True  # Make sure we test DETR components
    
    config.image_encoder = SimpleNamespace(
        name="siglip",
        use_pooler_output=False,
        model_path="",
    )

    # Set up necessary DETR properties
    config.detr_loss = True
    config.add_detr_layers = True
    config.num_query_tokens = 20
    
    # Other necessary configuration
    config.device = "cpu"
    config.batch_size = 2
    config.freeze_model = False
    config.use_precompute = False
    
    # Mock language model config
    config.language_model = SimpleNamespace(
        name="lmms-lab/llava-onevision-qwen2-0.5b-si"
    )

    # Initialize a model with random parameters
    logger.info("Creating first model")
    model1 = VisionLanguageModel(
        config=config,
        image_token_index=1,  # Use appropriate value
        num_new_tokens=10,    # Use appropriate value
        initializers=[[i] for i in range(10)]  # Example initializers
    )
    
    # Get parameters before saving
    logger.info("Capturing original parameters")
    original_proj_weight = model1.projector[0].weight.data.clone()
    original_detr_weight = model1.detr_integration.input_projection[0].weight.data.clone()
    print(model1.model.get_output_embeddings())
    original_llm_weight = model1.model.model.layers[0].self_attn.q_proj.weight.data.clone()
    original_output_embed_weight = model1.model.get_output_embeddings().trainable_lm_head.weight.data.clone()
    
    # Save state dict
    logger.info("Saving state dictionary")
    state_dict = model1.state_dict()

    #print("State dict keys:", state_dict.keys())  
    
    # Create a new model
    logger.info("Creating second model")
    model2 = VisionLanguageModel(
        config=config,
        image_token_index=1,
        num_new_tokens=10,
        initializers=[[i] for i in range(10)]
    )
    
    # Modify second model parameters to be different
    with torch.no_grad():
        model2.projector[0].weight.data += 1.0
        model2.detr_integration.input_projection[0].weight.data += 1.0
        model2.model.model.layers[0].self_attn.q_proj.weight.data += 1.0
        model2.model.get_output_embeddings().trainable_lm_head.weight.data += 1.0
    
    # Verify parameters are different
    logger.info("Verifying parameters are different")
    proj_diff1 = (model2.projector[0].weight.data - original_proj_weight).abs().sum().item()
    detr_diff1 = (model2.detr_integration.input_projection[0].weight.data - original_detr_weight).abs().sum().item()
    llm_diff1 = (model2.model.model.layers[0].self_attn.q_proj.weight.data - original_llm_weight).abs().sum().item()
    output_embed_diff1 = (model2.model.get_output_embeddings().trainable_lm_head.weight.data - original_output_embed_weight).abs().sum().item()

    logger.info(f"Difference before loading: Proj={proj_diff1:.4f}, DETR={detr_diff1:.4f}, LLM={llm_diff1:.4f}, OutputEmbed={output_embed_diff1:.4f}")
    
    assert proj_diff1 > 0, "Projector weights should be different before loading"
    assert detr_diff1 > 0, "DETR weights should be different before loading"
    assert llm_diff1 > 0, "LLM weights should be different before loading"
    assert output_embed_diff1 > 0, "Output embedding weights should be different before loading"
    
    # Load state dict into second model
    logger.info("Loading state dictionary")
    missing, unexpected = model2.load_state_dict(state_dict)
    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    # Verify parameters are now the same
    logger.info("Verifying parameters are now the same")
    proj_diff2 = (model2.projector[0].weight.data - original_proj_weight).abs().sum().item()
    detr_diff2 = (model2.detr_integration.input_projection[0].weight.data - original_detr_weight).abs().sum().item()
    llm_diff2 = (model2.model.model.layers[0].self_attn.q_proj.weight.data - original_llm_weight).abs().sum().item()
    output_embed_diff2 = (model2.model.get_output_embeddings().trainable_lm_head.weight.data - original_output_embed_weight).abs().sum().item()

    logger.info(f"Difference after loading: Proj={proj_diff2:.4f}, DETR={detr_diff2:.4f}, LLM={llm_diff2:.4f}, OutputEmbed={output_embed_diff2:.4f}")
    
    assert proj_diff2 < 1e-6, "Projector weights should be identical after loading"
    assert detr_diff2 < 1e-6, "DETR weights should be identical after loading"
    assert llm_diff2 < 1e-6, "LLM weights should be identical after loading"
    assert output_embed_diff2 < 1e-6, "Output embedding weights should be identical after loading"
    
    logger.info("State dict test passed successfully!")

if __name__ == "__main__":
    test_state_dict()