import io
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from django.core.files.base import ContentFile
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import redirect, render
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

HED_PROTO = Path("static/models/hed_deploy.prototxt")
HED_MODEL = Path("static/models/hed_pretrained_bsds.caffemodel")
_hed_net = None

from .forms import GalleryForm, ProjectForm, SignUpForm, UploadForm
from .models import GalleryPost, ProjectPost, SketchWork


def _suggest_outline_and_size(image):
    """Heuristically suggest an outline style and output size based on edges/contrast and dimensions."""
    gray = ImageOps.grayscale(image)
    gray_arr = np.asarray(gray, dtype=np.float32)
    edge_arr = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)

    edge_strength = edge_arr.mean()
    contrast = gray_arr.std()
    w, h = image.size
    longest = max(w, h)

    # Outline style: emphasize clean lines for detailed/high-contrast photos, trace for low-contrast, artistic otherwise.
    if edge_strength > 80 or contrast > 60:
        style = "clean"
    elif edge_strength < 35 and contrast < 35:
        style = "trace"
    else:
        style = "artistic"

    # Output size suggestion based on input resolution
    if longest >= 2000:
        size = "lg"
    elif longest >= 1400:
        size = "md"
    elif longest >= 900:
        size = "sm"
    else:
        size = "xs"

    return style, size


def _ai_refine(image, mode):
    """Lightweight outline-focused refinements (no external services)."""
    if mode == "clarity":
        # Extract strong edges, thicken, and blend for crisp pencil-like outlines
        gray = ImageOps.grayscale(image)
        edges = gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=0.4))
        edge_arr = np.asarray(edges, dtype=np.float32)
        cutoff = np.percentile(edge_arr, 70.0)
        lines = np.full_like(edge_arr, 255.0)
        lines[edge_arr >= cutoff] = 35.0
        line_img = Image.fromarray(lines.astype(np.uint8))
        line_img = line_img.filter(ImageFilter.MaxFilter(size=3)).filter(ImageFilter.UnsharpMask(radius=1.4, percent=260, threshold=1))
        edges_rgb = line_img.convert("RGB")
        return Image.blend(image, edges_rgb, alpha=0.5)
    if mode == "upscale":
        w, h = image.size
        target = (int(w * 1.25), int(h * 1.25))
        up = image.resize(target, Image.LANCZOS)
        outline = ImageOps.grayscale(up).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=0.4))
        outline_arr = np.asarray(outline, dtype=np.float32)
        cutoff = np.percentile(outline_arr, 70.0)
        lines = np.full_like(outline_arr, 255.0)
        lines[outline_arr >= cutoff] = 30.0
        outline_img = Image.fromarray(lines.astype(np.uint8))
        outline_img = outline_img.filter(ImageFilter.MaxFilter(size=3)).filter(ImageFilter.UnsharpMask(radius=1.2, percent=220, threshold=1))
        outline_rgb = outline_img.convert("RGB")
        # Edge-forward overlay after upscaling to keep outlines clear
        return Image.blend(up, outline_rgb, alpha=0.45)
    return image


def _edge_lines(gray, percentile=72.0, dark=35.0, maxfilter=3):
    """Create a crisp line mask from a grayscale image."""
    softened = gray.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.GaussianBlur(radius=0.6))
    edges = softened.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges, cutoff=1)
    edge_arr = np.asarray(edges, dtype=np.float32)
    threshold = np.percentile(edge_arr, percentile)
    lines = np.full_like(edge_arr, 255.0)
    lines[edge_arr >= threshold] = dark
    line_img = Image.fromarray(lines.astype(np.uint8))
    line_img = line_img.filter(ImageFilter.MaxFilter(size=maxfilter)).filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=220, threshold=1)
    )
    return line_img.convert("RGB")


def _opencv_trace(image):
    """OpenCV-based pencil trace (HED preferred when available)."""
    if cv2 is None:
        raise RuntimeError("OpenCV not available")

    global _hed_net
    use_hed = HED_PROTO.exists() and HED_MODEL.exists()
    if use_hed:
        if _hed_net is None:
            _hed_net = cv2.dnn.readNetFromCaffe(str(HED_PROTO), str(HED_MODEL))
        arr = np.array(image.convert("RGB"))[:, :, ::-1]
        blob = cv2.dnn.blobFromImage(arr, scalefactor=1.0, size=(arr.shape[1], arr.shape[0]),
                                     mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)
        _hed_net.setInput(blob)
        hed = _hed_net.forward()[0, 0]
        hed = cv2.resize(hed, (arr.shape[1], arr.shape[0]))
        hed = (255 * (1.0 - hed)).astype("uint8")
        # Light smooth, adaptive thresh to keep edges only
        hed = cv2.GaussianBlur(hed, (3, 3), 0)
        hed = cv2.adaptiveThreshold(hed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
        # Remove small blobs
        hed = cv2.morphologyEx(hed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # Slightly erode to slim strokes before thinning
        hed = cv2.erode(hed, np.ones((2, 2), np.uint8), iterations=1)
        # Thin lines
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            hed = cv2.ximgproc.thinning(hed)
        # Lighten lines further
        pil_line = Image.fromarray(hed).convert("L")
        pil_line = Image.blend(pil_line, Image.new("L", pil_line.size, 255), alpha=0.65)
        return pil_line.convert("RGB")

    # PIL RGB to BGR
    arr = np.array(image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=30, sigmaSpace=40)

    # Laplacian edges for faint pencil lines
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)

    # Normalize and invert
    lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
    inv = 255 - lap_norm

    # Adaptive threshold to keep only meaningful strokes
    th = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

    # Light dilation to reconnect, then thinning if available
    th = cv2.dilate(th, np.ones((2, 2), np.uint8), iterations=1)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        th = cv2.ximgproc.thinning(th)

    sketch = th
    return Image.fromarray(sketch).convert("RGB")


def _pencil_sketch(image_file, sketch_style="artistic", sketch_depth="medium", output_size="orig", ai_enhance="none"):
    """Convert an image file-like object to a pencil sketch PIL image (edge-first, minimal shading)."""
    image = Image.open(image_file).convert("RGB")
    image.thumbnail((1800, 1800))

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)

    # Preferred: OpenCV trace; fallback to PIL edge lines
    try:
        if cv2 is None:
            raise RuntimeError("OpenCV not available")
        trace = _opencv_trace(image)
    except Exception:
        trace = _edge_lines(gray, percentile=72.0, dark=30.0, maxfilter=3)

    # Style tuning
    if sketch_style == "trace":
        sketch = trace
        sketch_depth = "none"
    elif sketch_style == "artistic":
        lines = _edge_lines(gray, percentile=74.0, dark=32.0, maxfilter=3)
        sketch = Image.blend(trace, lines, alpha=0.8)
    else:
        # Clean: use trace directly
        sketch = trace

    # Minimal optional shading (disabled by default)
    shade_strength = {"none": 0.0, "light": 0.08, "medium": 0.14, "deep": 0.22}.get(sketch_depth, 0.0)
    if shade_strength > 0:
        tone = gray.filter(ImageFilter.GaussianBlur(radius=2.6))
        tone = ImageOps.autocontrast(tone, cutoff=2)
        tone_arr = np.asarray(tone, dtype=np.float32)
        sketch_arr = np.asarray(sketch.convert("L"), dtype=np.float32)
        shaded = np.clip((1 - 0.4 * shade_strength) * sketch_arr + shade_strength * tone_arr, 0, 255)
        shaded_img = Image.fromarray(shaded.astype(np.uint8))
        shaded_img = ImageOps.autocontrast(shaded_img, cutoff=1)
        sketch = shaded_img.convert("RGB")

    # Optional resizing
    size_map = {"orig": None, "lg": 1600, "md": 1024, "sm": 720, "xs": 480}
    target_max = size_map.get(output_size, None)
    if target_max:
        sketch.thumbnail((target_max, target_max))

    # Optional AI-like enhancement
    sketch = _ai_refine(sketch, ai_enhance)

    return sketch


def landing(request):
    return render(request, "landing.html")


def signup(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Welcome! Your account is ready.")
            return redirect("dashboard")
    else:
        form = SignUpForm()
    return render(request, "registration/signup.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("landing")


@login_required
def dashboard(request):
    tab = request.GET.get("tab", "upload")
    form = UploadForm()
    gallery_form = GalleryForm()
    project_form = ProjectForm()
    result_url = None

    if request.method == "POST":
        tab = request.POST.get("tab", "upload")
        if tab == "gallery":
            gallery_form = GalleryForm(request.POST, request.FILES)
            if gallery_form.is_valid():
                GalleryPost.objects.create(
                    user=request.user,
                    title=gallery_form.cleaned_data["title"],
                    body=gallery_form.cleaned_data["body"],
                    image=gallery_form.cleaned_data["image"],
                )
                messages.success(request, "Posted to gallery.")
                tab = "gallery"
            else:
                messages.error(request, "Please provide a title and image for the gallery post.")
        elif tab == "project":
            project_form = ProjectForm(request.POST, request.FILES)
            if project_form.is_valid():
                ProjectPost.objects.create(
                    user=request.user,
                    title=project_form.cleaned_data["title"],
                    body=project_form.cleaned_data["body"],
                    image=project_form.cleaned_data.get("image"),
                )
                messages.success(request, "Project post published.")
                tab = "project"
            else:
                messages.error(request, "Please provide a title for the project post.")
        else:
            form = UploadForm(request.POST, request.FILES)
            if form.is_valid():
                upload = form.cleaned_data["image"]
                upload_content = upload.read()

                work = SketchWork(user=request.user)

                # Suggest outline style and output size from the uploaded image
                suggest_style, suggest_size = _suggest_outline_and_size(Image.open(io.BytesIO(upload_content)))

                sketch_image = _pencil_sketch(
                    io.BytesIO(upload_content),
                    sketch_style=suggest_style,
                    sketch_depth=form.cleaned_data["sketch_depth"],
                    output_size=suggest_size,
                    ai_enhance=form.cleaned_data.get("ai_enhance", "none"),
                )
                buffer = io.BytesIO()
                sketch_image.save(buffer, format="PNG", optimize=True)
                buffer.seek(0)
                work.sketch_image.save(f"sketch_{uuid4().hex}.png", ContentFile(buffer.getvalue()), save=False)
                work.save()

                result_url = work.sketch_image.url
                messages.success(request, "Sketch generated successfully.")
                messages.info(request, f"AI suggestion applied: outline '{suggest_style}' at size '{suggest_size}'.")
                tab = "upload"
            else:
                messages.error(request, "Please upload a valid image file.")

    recent_qs = SketchWork.objects.filter(user=request.user).order_by("-created_at")
    q = request.GET.get("q", "").strip()
    start_date = request.GET.get("start_date")
    end_date = request.GET.get("end_date")

    if q:
        recent_qs = recent_qs.filter(sketch_image__icontains=q)
    if start_date:
        try:
            dt = datetime.fromisoformat(start_date)
            recent_qs = recent_qs.filter(created_at__date__gte=dt.date())
        except ValueError:
            pass
    if end_date:
        try:
            dt = datetime.fromisoformat(end_date)
            recent_qs = recent_qs.filter(created_at__date__lte=dt.date())
        except ValueError:
            pass

    recent = list(recent_qs[:30])
    recent_gallery = list(GalleryPost.objects.filter(user=request.user)[:20])
    recent_projects = list(ProjectPost.objects.filter(user=request.user)[:20])

    return render(
        request,
        "sketch/dashboard.html",
        {
            "form": form,
            "gallery_form": gallery_form,
            "project_form": project_form,
            "result_url": result_url,
            "recent": recent,
            "recent_gallery": recent_gallery,
            "recent_projects": recent_projects,
            "tab": tab,
            "filters": {"q": q, "start_date": start_date or "", "end_date": end_date or "", "size": request.GET.get("size", "orig")},
        },
    )


def section(request, slug):
    sections = {
        "create": {
            "title": "Create",
            "lead": "Bring images in, jumpstart sketches, or explore ready-made templates.",
            "items": [
                "Import Artwork / Capture Photo",
                "Procedural Sketch Generator",
                "Template Gallery",
            ],
        },
        "editor": {
            "title": "Editor",
            "lead": "Full-featured canvas with layers, brushes, masks, and precise transforms.",
            "items": [
                "Canvas Workspace",
                "Layers",
                "Brushes & Tools",
                "Masks & Local Editing",
                "Transform Tools",
            ],
        },
        "ai-enhancement": {
            "title": "AI Enhancement",
            "lead": "Enhance, stylize, refine, and upscale your sketches with AI tooling.",
            "items": [
                "Enhance Artwork (Image-to-Image)",
                "Shading & Detail Refinement",
                "AI Illusion Generation",
                "Style Transfer",
                "Super-Resolution Upscaling",
                "Colorization",
            ],
        },
        "projects": {
            "title": "Projects",
            "lead": "Stay organized with versions, autosaves, and shared collaborations.",
            "items": ["My Projects", "Version History", "Autosaved Drafts", "Shared Collaborations"],
        },
        "community": {
            "title": "Gallery",
            "lead": "Share illusions and AR-ready pieces—upload progress or finished work as photos or videos.",
            "items": [
                "Progress work (photo or video uploads)",
                "Finished work (photo or video uploads)",
                "Illusion showcase",
                "Augmented Reality (AR) experiences",
            ],
        },
        "account": {
            "title": "Account",
            "lead": "Manage your profile, presets, preferences, billing, and sessions.",
            "items": ["My Profile", "Saved Styles / Presets", "Settings", "Billing / Credits", "Sign Out"],
        },
    }
    section_data = sections.get(slug)
    if not section_data:
        section_data = {"title": "Hybrid Artistic Engine", "lead": "Discover what you can create.", "items": []}
    return render(request, "sketch/section.html", {"section": section_data})


@login_required
def download_all(request):
    # Apply same filters for consistency
    recent_qs = SketchWork.objects.filter(user=request.user).order_by("-created_at")
    q = request.GET.get("q", "").strip()
    start_date = request.GET.get("start_date")
    end_date = request.GET.get("end_date")

    if q:
        recent_qs = recent_qs.filter(sketch_image__icontains=q)
    if start_date:
        try:
            dt = datetime.fromisoformat(start_date)
            recent_qs = recent_qs.filter(created_at__date__gte=dt.date())
        except ValueError:
            pass
    if end_date:
        try:
            dt = datetime.fromisoformat(end_date)
            recent_qs = recent_qs.filter(created_at__date__lte=dt.date())
        except ValueError:
            pass

    buffer = io.BytesIO()
    import zipfile

    # Handle optional dimension scaling
    size_code = request.GET.get("size", "orig")
    size_map = {
        "orig": None,
        "lg": 1600,
        "md": 1024,
        "sm": 720,
        "xs": 480,
    }
    target_max = size_map.get(size_code, None)

    def _resize_if_needed(image_file, filename):
        if not target_max:
            return filename, image_file.read()
        with Image.open(image_file) as img:
            img = img.convert("RGB")
            img.thumbnail((target_max, target_max))
            buffer_inner = io.BytesIO()
            img.save(buffer_inner, format="PNG", optimize=True)
            buffer_inner.seek(0)
            name_parts = filename.rsplit(".", 1)
            sized_name = f"{name_parts[0]}_{size_code}.png"
            return sized_name, buffer_inner.getvalue()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for work in recent_qs:
            filename = work.sketch_image.name.split("/")[-1]
            with work.sketch_image.open("rb") as fh:
                out_name, out_bytes = _resize_if_needed(fh, filename)
                zf.writestr(out_name, out_bytes)
    buffer.seek(0)
    response = HttpResponse(buffer.getvalue(), content_type="application/zip")
    response["Content-Disposition"] = 'attachment; filename="sketches.zip"'
    return response


def gallery(request):
    posts = GalleryPost.objects.select_related("user").order_by("-created_at")[:60]
    return render(request, "sketch/gallery.html", {"posts": posts})


def projects_view(request):
    posts = ProjectPost.objects.select_related("user").order_by("-created_at")[:60]
    return render(request, "sketch/projects.html", {"posts": posts})


@login_required
def user_detail_admin(request, user_id):
    if not request.user.is_staff:
        return redirect("dashboard")
    User = get_user_model()
    user_obj = User.objects.filter(id=user_id).first()
    if not user_obj:
        return redirect("admin_dashboard")

    sketches = SketchWork.objects.filter(user=user_obj).order_by("-created_at")
    gallery_posts = GalleryPost.objects.filter(user=user_obj).order_by("-created_at")
    project_posts = ProjectPost.objects.filter(user=user_obj).order_by("-created_at")

    stats = {
        "sketches": sketches.count(),
        "gallery": gallery_posts.count(),
        "projects": project_posts.count(),
    }

    return render(
        request,
        "sketch/user_detail.html",
        {
            "user_obj": user_obj,
            "sketches": sketches,
            "gallery_posts": gallery_posts,
            "project_posts": project_posts,
            "stats": stats,
        },
    )


@login_required
def admin_dashboard(request):
    if not request.user.is_staff:
        return redirect("dashboard")

    project_form = ProjectForm()
    gallery_form = GalleryForm()
    User = get_user_model()

    if request.method == "POST":
        if request.POST.get("form_type") == "project":
            project_form = ProjectForm(request.POST, request.FILES)
            if project_form.is_valid():
                ProjectPost.objects.create(
                    user=request.user,
                    title=project_form.cleaned_data["title"],
                    body=project_form.cleaned_data["body"],
                    image=project_form.cleaned_data.get("image"),
                )
                messages.success(request, "Project post published.")
        elif request.POST.get("form_type") == "gallery":
            gallery_form = GalleryForm(request.POST, request.FILES)
            if gallery_form.is_valid():
                GalleryPost.objects.create(
                    user=request.user,
                    title=gallery_form.cleaned_data["title"],
                    body=gallery_form.cleaned_data["body"],
                    image=gallery_form.cleaned_data["image"],
                )
                messages.success(request, "Gallery post published.")

    total_users = User.objects.count()
    total_sketches = SketchWork.objects.count()
    total_gallery = GalleryPost.objects.count()
    total_projects = ProjectPost.objects.count()

    recent_users = User.objects.order_by("-date_joined")[:10]
    recent_sketches = SketchWork.objects.select_related("user").order_by("-created_at")[:10]
    recent_posts = ProjectPost.objects.select_related("user").order_by("-created_at")[:10]
    all_sketches = SketchWork.objects.select_related("user").order_by("-created_at")[:100]
    all_gallery = GalleryPost.objects.select_related("user").order_by("-created_at")[:100]

    stats = {
        "total_users": total_users,
        "total_sketches": total_sketches,
        "total_gallery": total_gallery,
        "total_projects": total_projects,
    }

    return render(
        request,
        "sketch/admin_dashboard.html",
        {
            "project_form": project_form,
            "gallery_form": gallery_form,
            "recent_users": recent_users,
            "recent_sketches": recent_sketches,
            "recent_posts": recent_posts,
            "stats": stats,
            "all_sketches": all_sketches,
            "all_gallery": all_gallery,
        },
    )
