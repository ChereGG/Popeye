import {Component, OnInit, ViewChild} from '@angular/core';
import {NgxChessBoardComponent, NgxChessBoardView} from 'ngx-chess-board';
import {BoardService} from "../../services/board.service";
import {HttpClient} from "@angular/common/http";

@Component({
  selector: 'app-chess-board',
  templateUrl: './chess-board.component.html',
  styleUrls: ['./chess-board.component.css']
})
export class ChessBoardComponent implements OnInit {
  private i : number =0;
  @ViewChild(NgxChessBoardComponent) board: NgxChessBoardView;

  public constructor(private boardService: BoardService){

  }
  ngOnInit(): void {

  }

  ngAfterViewInit(): void {
  }

  undo(): void {
    this.board.undo();
  }

  nextMove(): void{
    if(this.i%2 == 0) {
      const fen = this.board.getFEN();
      this.boardService.getNextMove(fen).subscribe(response => {
        this.board.move(response['move']);
      });
    }
    this.i++;
  }
}
