import SimpleITK as sitk

reader = sitk.ImageFileReader()
reader.SetImageIO("TIFFImageIO")
inputImageFileName = r'E:\HE+CAM5\tiff\new1.tiff'
reader.SetFileName(inputImageFileName)
image = reader.Execute()
print(image.shape)
