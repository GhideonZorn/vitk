import itk
import vtk
import os
import ipywidgets as widgets
from IPython.display import display
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

# Constants
PixelType = itk.D
Dimension = 3
ImageType = itk.Image[PixelType, Dimension]

# Data
data1 = "data/case6_gre1.nrrd"
data2 = "data/case6_gre2.nrrd"

# Images
image1 = itk.imread(data1, PixelType)
image2 = itk.imread(data2, PixelType)

#####   Utilitaire   #####
def find_number_of_files(regex: str):
    files = [file for file in os.listdir("results") if file.startswith(regex)]
    return len(files)

index_transformed = find_number_of_files("transformed")
index_segmented = find_number_of_files("segmented")
index_visualisation = find_number_of_files("visualisation")
index_combined = find_number_of_files("combined")

#####   Recalage   #####

# Preprocessing
def preprocess_v1(image):
    image = itk.median_image_filter(image, radius=1)

    resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
    resampler.SetInput(image)
    resampler.SetSize(image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.Update()

    image = resampler.GetOutput()

    return image

def preprocess_v2(image):
    print("Starting preprocessing...")
    # Smoothing
    smoothing = itk.CurvatureFlowImageFilter[ImageType, ImageType].New(
        Input=image,
        TimeStep=0.125,
        NumberOfIterations=5,
    )
    smoothing.Update()

    # Rescaling
    rescaling = itk.RescaleIntensityImageFilter[ImageType, ImageType].New(
        Input=smoothing.GetOutput(),
        OutputMinimum=0,
        OutputMaximum=255,
    )
    rescaling.Update()

    print("Preprocessing completed.")
    return rescaling.GetOutput()

# Registration
def register(image1, image2):
    print("Starting registration...")
    dimension = image1.GetImageDimension()
    TransformType = itk.TranslationTransform[PixelType, dimension]

    transform = TransformType.New()
    transform.SetIdentity()

    metric = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType].New()

    # Optimizer
    optimizer = itk.RegularStepGradientDescentOptimizerv4[PixelType].New(
        LearningRate=4,
        MinimumStepLength=0.001,
        RelaxationFactor=0.5,
        NumberOfIterations=200,
    )

    # interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()

    registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New(
        Metric=metric,
        Optimizer=optimizer,
        # Interpolator=interpolator,
        InitialTransform=transform,
        FixedImage=image1,
        MovingImage=image2,
    )
    registration.Update()

    print("Registration completed.")
    return registration, optimizer

# Transformation
def transform_image(image1, image2):
    registration, optimizer = register(image1, image2)

    print("Starting image transformation...")
    # Resample
    transformed_image = itk.resample_image_filter(
        image2,
        transform=registration.GetTransform(),
        use_reference_image=True,
        reference_image=image1,
    )

    itk.imwrite(transformed_image, f"results/transformed_{index_transformed}.nrrd")
    print("Image transformation completed.")

    final_parameters = registration.GetOutput().Get().GetParameters()
    final_number_of_iterations = optimizer.GetCurrentIteration()
    final_metric_value = optimizer.GetCurrentMetricValue()

    # report = f"Final parameters =\n{final_parameters.GetElement(0)},\
    #         \n{final_parameters.GetElement(1)},\n{final_parameters.GetElement(2)},\
    #         \n{final_parameters.GetElement(3)},\n{final_parameters.GetElement(4)},\
    #         \n{final_parameters.GetElement(5)}\
    #         \nNumber of iterations = {final_number_of_iterations}\
    #         \nMetric value = {final_metric_value}"
    report = f"Number of iterations = {final_number_of_iterations}\
            \nMetric value = {final_metric_value}"

    return transformed_image, report

#####   Segmentation   #####

def segment_image(image, lower, upper):
    print("Starting image segmentation...")

    # The two seed of the two tumor
    seed_index1 = itk.Index[3]()
    seed_index1.SetElement(0, 125)
    seed_index1.SetElement(1, 65)
    seed_index1.SetElement(2, 80)

    seed_index2 = itk.Index[3]()
    seed_index2.SetElement(0, 100)
    seed_index2.SetElement(1, 80)
    seed_index2.SetElement(2, 80)

    # Thresholding the image with the given boundaries
    segmented = itk.ConnectedThresholdImageFilter[ImageType, ImageType].New(
        Lower=lower,
        Upper=upper,
        ReplaceValue=255,
    )

    segmented.AddSeed(seed_index1)
    segmented.AddSeed(seed_index2)
    segmented.SetInput(image)
    segmented.Update()
    segmented_image = segmented

    itk.imwrite(segmented_image, f"results/segmented_{index_segmented}.nrrd")
    print("Image segmentation completed.")
    return segmented_image

def render_nrrd_image(path: str):
    print("Rendering NRRD image...")
    # Lire une image nrrd avec vtk
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(path)
    reader.Update()

    # Créer un contour
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(reader.GetOutputPort())
    contour.SetValue(0, 131)

    # Créer un mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())

    # Créer un acteur
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Créer un renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.5, 0.5, 0.5)

    # Créer une fenêtre
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(800, 800)

    # Créer un interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    # Créer un style
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    # Afficher la fenêtre
    window.Render()
    interactor.Start()
    print("Rendering completed.")

def render_tumor(brain, tumor):
    print("Rendering tumor...")
    # Lire une image nrrd avec vtk
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(brain)
    reader.Update()

    # Créer un contour
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(reader.GetOutputPort())
    contour.SetValue(0, 135)

    # Créer un mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()

    # Créer un acteur
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    property = vtk.vtkProperty()
    property.SetColor(0, 0, 1)  # Set blue color for the brain contour
    actor.SetProperty(property)

    # Lire une image nrrd avec vtk
    reader2 = vtk.vtkNrrdReader()
    reader2.SetFileName(tumor)
    reader2.Update()

    # Créer un contour
    contour2 = vtk.vtkContourFilter()
    contour2.SetInputConnection(reader2.GetOutputPort())
    contour2.SetValue(0, 135)

    # Créer un mapper
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputConnection(contour2.GetOutputPort())
    mapper2.ScalarVisibilityOff()

    # Créer un acteur
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetColor(1, 0, 0)  # Set red color for the tumor contour

    # Déplacer le tumor dans le cerveau
    transform = vtk.vtkTransform()
    transform.Translate(-83, 126, -252)
    actor2.SetUserTransform(transform)

    # Créer un renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor(actor2)
    renderer.SetBackground(0.5, 0.5, 0.5)

    # Créer une fenêtre
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(800, 800)

    # Créer un interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    # Créer un style
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    # Afficher la fenêtre
    window.Render()
    interactor.Start()
    print("Tumor rendering completed.")

# def cut_brain_at_slice_v1(image, slice_index=81):
#     print("Cutting brain at slice...")
#     # Lire une image nrrd avec vtk
#     reader = vtk.vtkNrrdReader()
#     reader.SetFileName(image)
#     reader.Update()
#     # Créer un extracteur
#     extract = vtk.vtkExtractVOI()
#     extract.SetInputConnection(reader.GetOutputPort())
#     extract.SetVOI(0, 255, 0, 255, slice_index, slice_index)
#     extract.Update()

#     # Créer un mapper
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputConnection(extract.GetOutputPort())

#     # Créer un acteur
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)

#     # Créer un renderer
#     renderer = vtk.vtkRenderer()
#     renderer.AddActor(actor)
#     renderer.SetBackground(0.5, 0.5, 0.5)

#     # Créer une fenêtre
#     window = vtk.vtkRenderWindow()
#     window.AddRenderer(renderer)
#     window.SetSize(800, 800)

#     # Créer un interactor
#     interactor = vtk.vtkRenderWindowInteractor()
#     interactor.SetRenderWindow(window)

#     # Créer un style
#     style = vtk.vtkInteractorStyleTrackballCamera()
#     interactor.SetInteractorStyle(style)

#     # Afficher la fenêtre
#     window.Render()
#     interactor.Start()
#     print("Brain cutting completed.")

def cut_brain_at_slice_v2(image, slice_index=81):
    print("Cutting brain at slice...")
    # Convert itk image to numpy array
    array = itk.array_from_image(image)

    # Cut the brain at the slice index
    array = array[slice_index:175, :, :]

    # Convert numpy array to itk image
    cut_brain = itk.image_from_array(array)

    itk.imwrite(cut_brain, f"results/cut_brain.nrrd")
    print("Brain cut completed.")
    return cut_brain

#####   Main   #####

# transformed_image = transform_image(image1, image2)
transformed_image = None
segmented_image = None

def plot_slices(image1, image2, transformed_image, segmented_image, index=81):
    fixed_view = itk.array_view_from_image(image1)
    moving_view = itk.array_view_from_image(image2)
    transform_view = itk.array_view_from_image(transformed_image)
    segmented_view = itk.array_view_from_image(segmented_image)

    fixed = index
    moving = index
    transformed = index
    segmented = index
    plt.figure(figsize=(10, 6.8))

    plt.subplot(2, 3, 1)
    plt.imshow(fixed_view[fixed, :, :], interpolation_stage='rgba')
    plt.title("Fixed image")

    plt.subplot(2, 3, 2)
    plt.imshow(moving_view[moving, :, :], interpolation_stage='rgba')
    plt.title("Moving image")

    plt.subplot(2, 3, 3)
    plt.imshow(transform_view[transformed, :, :], interpolation_stage='rgba')
    plt.title("Transformed image")

    plt.subplot(2, 3, 5)
    plt.imshow(segmented_view[segmented, :, :], cmap='gray')
    plt.title("Segmented image")

    plt.savefig(f"results/visualisation.png")
    plt.close()

# render_nrrd_image(f"results/transformed_{index_transformed}.nrrd")

def analysis(transformed_image, segmented_image):
    print("Starting analysis...")
    segmented_slice = itk.array_view_from_image(segmented_image)
    transformed_slice = itk.array_view_from_image(transformed_image)

    masked_slice = np.ma.masked_where(segmented_slice == 0, transformed_slice)
    box_slice = [slice(69, 89, None), slice(56, 77, None), slice(103, 141, None)]
    box_mask = np.zeros(masked_slice.shape, dtype=bool)
    box_mask[box_slice[0], box_slice[1], box_slice[2]] = True
    boxed_tumor_slices = np.ma.masked_where(masked_slice, ~box_mask).filled(0)
    # itk.imwrite(itk.image_view_from_array(boxed_tumor_slices), "results/tumor2.nrrd")

    # Too much noise in this slice
    transformed_slice = transformed_slice[:87]
    boxed_tumor_slices = boxed_tumor_slices[:87]

    augmented_volume = 100 * np.count_nonzero(boxed_tumor_slices) / np.count_nonzero(transformed_slice)
    augmented_intensity = 100 * int(boxed_tumor_slices.sum(axis=1).sum()) / int(transformed_slice.sum(axis=1).sum())

    report_analysis = f"Augmented volume = {augmented_volume}%\nAugmented intensity = {augmented_intensity}%"

    return report_analysis

#####   Dashboard   #####

dashboard = tk.Tk()
dashboard.title("Tumor Segmentation")
dashboard.geometry("1500x1100")

print("Setup dashboard...")

# Recalage frame
recalage_frame = tk.Frame(dashboard)
recalage_frame.config(bd=1, relief=tk.SOLID)
recalage_frame.pack(padx=5, pady=5)

# Field to display the transformation report
transformation_report = tk.Text(recalage_frame, height=3, width=40)
transformation_report.insert(tk.END, "Transformation report will be displayed here.")
transformation_report.config(state="disabled")

# Transformation button
def transform_button_clicked():
    global transformed_image, image1, image2
    print("Transformation button clicked")

    image1 = preprocess_v1(image1)
    image2 = preprocess_v1(image2)
    #image1 = preprocess_v2(image1)
    #image2 = preprocess_v2(image2)
    transformed_image, report = transform_image(image1, image2)

    print(report)
    transformation_report.config(state="normal")
    transformation_report.delete(1.0, tk.END)
    transformation_report.insert(tk.END, report)

    segmentation_button.config(state="normal")

transform_button = tk.Button(recalage_frame, text="Registration", command=transform_button_clicked)

# Segmentation button
def segmentation_button_clicked():
    global segmented_image
    print("Segmentation button clicked")
    segmented_image = segment_image(transformed_image, 450, 800)
    visualisation_button.config(state="normal")

# Segmentation frame
segmentation_frame = tk.Frame(dashboard)
segmentation_frame.config(bd=1, relief=tk.SOLID)
segmentation_frame.pack(padx=5, pady=5)

segmentation_button = tk.Button(segmentation_frame, text="Segmentation", command=segmentation_button_clicked)

# segmentation_button.config(state="disabled")

# Image display
image_canvas = tk.Canvas(recalage_frame, width=1000, height=600)

# Update image display
def update_image_display(image_path):
    image = tk.PhotoImage(file=image_path)
    image_canvas.create_image(0, 0, anchor=tk.NW, image=image)
    image_canvas.config(width=image.width(), height=image.height())
    image_canvas.image = image

# Slice selection slider
slice_slider = tk.Scale(
    recalage_frame,
    from_=0,
    to=175,
    orient=tk.HORIZONTAL,
    label="Select Slice to Visualise",
    length=300,
    resolution=1,
)
slice_slider.set(81)

# Update slice index
def update_slice_index(value):
    index = int(value)
    plot_slices(image1, image2, transformed_image, segmented_image, index)
    update_image_display("results/visualisation.png")

# Bind slice selection to update function
slice_slider.config(command=update_slice_index)

# Visualisation button
def visualisation_recalage_button_clicked():
    global transformed_image
    global segmented_image

    if transformed_image is None:
        transformed_image = itk.imread(f"results/transformed_{index_transformed - 1}.nrrd", PixelType)

    if segmented_image is None:
        # FIXME: dislay a black image insted of the image1
        segmented_image = image1

    # Change the slice index
    slice_slider.set(81)

    print("Visualisation for recalage button clicked")
    # Change the slice index
    plot_slices(image1, image2, transformed_image, segmented_image, slice_slider.get())

    update_image_display("results/visualisation.png")

visualisation_recalage_button = tk.Button(recalage_frame, text="Visualisation", command=visualisation_recalage_button_clicked)
# visualisation_button.config(state="disabled")

# Visualisation frame
visualisation_frame = tk.Frame(dashboard)
visualisation_frame.config(bd=1, relief=tk.SOLID)
visualisation_frame.pack(padx=5, pady=5)

# Visualisation button
def visualisation_button_clicked():
    print("Visualisation button clicked")
    # combine_image = combine_images(transformed_image, segmented_image)
    # itk.imwrite(f"results/combined_{index_combined}.nrrd", combine_image)
    # render_nrrd_image(combine_image)
    cut_brain_at_slice_v2(transformed_image)
    # itk.imwrite(cut_brain, f"results/cut_brain.nrrd")
    # render_nrrd_image(f"results/cut_brain.nrrd")
    brain = f"results/cut_brain.nrrd"
    tumor = f"results/segmented_{index_segmented}.nrrd"
    render_tumor(brain, tumor)

visualisation_button = tk.Button(visualisation_frame, text="Visualisation", command=visualisation_button_clicked)

# Analysis button
def analysis_button_clicked():
    print("Analysis button clicked")
    report = analysis(image1, segmented_image)
    analysis_report.config(state="normal")
    analysis_report.delete(1.0, tk.END)
    # analysis_report.insert(tk.END, report)

analysis_button = tk.Button(visualisation_frame, text="Analysis", command=analysis_button_clicked)

# Text item to display the analysis report
analysis_report = tk.Text(visualisation_frame, height=2, width=40)
analysis_report.insert(tk.END, "Analysis report will be displayed here.")
analysis_report.config(state="disabled")

# Run everything button
def run_everything_button_clicked():
    transform_button_clicked()
    segmentation_button_clicked()
    visualisation_recalage_button_clicked()
    analysis_button_clicked()
    visualisation_button_clicked()

run_everything_button = tk.Button(text="Run Everything", command=run_everything_button_clicked)

# Quit button
def quit_button_clicked():
    dashboard.quit()

quit_button = tk.Button(text="Quit", command=quit_button_clicked)

# Dashboard layout
recalage_frame.pack()

transform_button.pack()
transformation_report.pack()
visualisation_recalage_button.pack()
slice_slider.pack()
image_canvas.pack()

segmentation_frame.pack()
segmentation_button.pack()

visualisation_frame.pack()
visualisation_button.pack()

analysis_button.pack()
analysis_report.pack()

run_everything_button.pack()

quit_button.pack()

print("Dashboard setup completed.")

dashboard.mainloop()
