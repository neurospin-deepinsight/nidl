import unittest
from nidl.datasets.openbhb import OpenBHB
from nidl.models import JEPA
from nidl.volume.transforms import CropOrPad
from nidl.data.collate import MaskCollateFunction
from torch.utils.data import DataLoader


class TestJEPA(unittest.TestCase):

    def test_load_jepa(self):
        jepa = JEPA(
            encoder="vit_tiny_3d",
            encoder_kwargs={"pool": None},
            predictor_kwargs={"predictor_embed_dim": 192, "num_heads": 3, "depth": 6},
            lr_scheduler=None,
            num_sanity_val_steps=0
        )
    
    def test_run_jepa(self):
        # Load the data
        dataset_train = OpenBHB("/neurospin/signatures/bd261576/openBHB", 
                                target="age", modality="quasiraw", split="train", 
                                transforms=CropOrPad((1, 128, 128, 128)))
        dataset_test = OpenBHB("/neurospin/signatures/bd261576/openBHB", 
                               target="age", modality="quasiraw", split="val",
                            transforms=CropOrPad((1, 128, 128, 128)))
        train_dataloader = DataLoader(dataset_train, batch_size=4, num_workers=15, shuffle=True, 
                                    collate_fn=MaskCollateFunction())
        val_dataloader = DataLoader(dataset_test, batch_size=4, num_workers=10, shuffle=False,
                                    collate_fn=MaskCollateFunction())
        jepa = JEPA(
            encoder="vit_tiny_3d",
            encoder_kwargs={"pool": None},
            predictor_kwargs={"predictor_embed_dim": 192, "num_heads": 3, "depth": 6},
            lr_scheduler=None,
            num_sanity_val_steps=0
        )

        jepa.fit(train_dataloader, val_dataloader)
    
if __name__ == '__main__':
    unittest.main()