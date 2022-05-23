import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { NgxChessBoardModule } from 'ngx-chess-board';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ChessBoardComponent } from './components/chess-board/chess-board.component';
import {HttpClient, HttpClientModule} from "@angular/common/http";
import { FrontPageComponent } from './components/front-page/front-page.component';

@NgModule({
  declarations: [
    AppComponent,
    ChessBoardComponent,
    FrontPageComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    NgxChessBoardModule.forRoot(),
    HttpClientModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
