from django.conf.urls.static import static
from django.urls import path

from popeye import views
from PopeyeBackend import settings

urlpatterns = [
    path('api/next-move', views.next_move),
]
