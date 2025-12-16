from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
    path("", views.landing, name="landing"),
    path("app/", views.dashboard, name="dashboard"),
    path("app/download-all/", views.download_all, name="download_all"),
    path("signup/", views.signup, name="signup"),
    path("login/", auth_views.LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("create/", views.section, {"slug": "create"}, name="create"),
    path("editor/", views.section, {"slug": "editor"}, name="editor"),
    path("ai-enhancement/", views.section, {"slug": "ai-enhancement"}, name="ai-enhancement"),
    path("projects/", views.projects_view, name="projects"),
    path("community/", views.gallery, name="gallery"),
    path("account/", views.section, {"slug": "account"}, name="account"),
    path("gallery/", views.gallery, name="gallery"),
    path("admin-panel/", views.admin_dashboard, name="admin_dashboard"),
    path("admin-panel/user/<int:user_id>/", views.user_detail_admin, name="admin_user_detail"),
]
