from workspace_generation_utils import *
from image_cache import ImageCache
import os

TOTAL_WORKSPACES = 10000
OUTPUT_DIR = "scenario_params/vision_harder"


if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

generator = WorkspaceGenerator(obstacle_count_probabilities={2: 0.05, 3: 0.5, 4: 0.4, 5: 0.05})
for i in range(TOTAL_WORKSPACES):
    save_path = os.path.join(OUTPUT_DIR, '{}_workspace.pkl'.format(i))

    if os.path.exists(save_path):
        print("workspace %d already exists" % i)
        continue

    print("generateing workspace %d" % i)
    workspace_params = generator.generate_workspace()
    workspace_params.save(save_path)

print("Creating Image Cache")
ImageCache(OUTPUT_DIR, True)
