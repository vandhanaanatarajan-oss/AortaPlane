import SimpleITK as sitk

case = "SEGA_D1"
orig = sitk.ReadImage(f"/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/input/{case}_0000.nii.gz")
pred = sitk.ReadImage(f"/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/output/{case}.nii.gz")

print("Original size:", orig.GetSize(), "spacing:", orig.GetSpacing())
print("Predicted size:", pred.GetSize(), "spacing:", pred.GetSpacing())
print("Same size?", orig.GetSize() == pred.GetSize())
