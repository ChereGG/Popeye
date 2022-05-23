import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {ChessBoardComponent} from './components/chess-board/chess-board.component';
import {FrontPageComponent} from './components/front-page/front-page.component';
const routes: Routes = [
  { path: 'game', component: ChessBoardComponent},
  { path: '', component: FrontPageComponent},

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
