import {Component, OnInit, ViewChild} from '@angular/core';
import {NgxChessBoardComponent, NgxChessBoardService} from 'ngx-chess-board';
import {NgxChessBoardView} from 'ngx-chess-board';
import {KeyedRead} from "@angular/compiler";

@Component({
  selector: 'app-chess-board',
  templateUrl: './chess-board.component.html',
  styleUrls: ['./chess-board.component.css']
})
export class ChessBoardComponent implements OnInit {

  @ViewChild(NgxChessBoardComponent) board: NgxChessBoardView;

  constructor() {

  }

  ngOnInit(): void {
  }

  ngAfterViewInit(): void {
  }

  undo(): void {
    this.board.undo();
  }
}
