from django.conf.urls.static import static
from django.urls import path

from popeye import views
from PopeyeBackend import settings

urlpatterns = [
    path('api/send-move-reinforcement', views.send_move_reinforcement),
    path('api/send-move-supervised', views.send_move_supervised),
    path('api/undo-reinforcement', views.undo_reinforcement),
]
