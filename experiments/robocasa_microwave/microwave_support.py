from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
class OfflineRunner(BaseImageRunner):
    """Keep simulator rollouts separate from the official training workspace."""
    def run(self, policy):
        return {}
