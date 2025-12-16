from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class UploadForm(forms.Form):
    image = forms.ImageField(label="Upload an image to sketch")
    sketch_style = forms.ChoiceField(
        choices=[("artistic", "Artistic pencil work"), ("clean", "Clean technical lines"), ("trace", "Trace/outline")],
        initial="artistic",
        label="Sketch style",
    )
    sketch_depth = forms.ChoiceField(
        choices=[
            ("none", "No depth"),
            ("light", "Light depth"),
            ("medium", "Balanced depth"),
            ("deep", "Rich depth"),
        ],
        initial="none",
        label="Sketch depth",
    )
    output_size = forms.ChoiceField(
        choices=[
            ("orig", "Original dimensions"),
            ("lg", "Large (max 1600px)"),
            ("md", "Medium (max 1024px)"),
            ("sm", "Small (max 720px)"),
            ("xs", "Compact (max 480px)"),
        ],
        initial="orig",
        label="Output size",
    )
    ai_enhance = forms.ChoiceField(
        choices=[
            ("none", "None"),
            ("clarity", "AI outline clarity"),
            ("upscale", "AI outline upscaler"),
        ],
        initial="none",
        label="AI enhancement",
    )


class GalleryForm(forms.Form):
    title = forms.CharField(max_length=180, label="Title")
    body = forms.CharField(label="Caption", required=False, widget=forms.Textarea(attrs={"rows": 3}))
    image = forms.ImageField(label="Artwork image")


class ProjectForm(forms.Form):
    title = forms.CharField(max_length=180, label="Title")
    body = forms.CharField(label="Body", required=False, widget=forms.Textarea(attrs={"rows": 4}))
    image = forms.ImageField(label="Cover image", required=False)


class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=False)
    last_name = forms.CharField(max_length=30, required=False)

    class Meta:
        model = User
        fields = ("username", "email", "first_name", "last_name", "password1", "password2")
