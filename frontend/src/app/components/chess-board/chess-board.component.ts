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
  @ViewChild(NgxChessBoardComponent) board: NgxChessBoardView;

  private send:boolean=true;


  public constructor(private boardService: BoardService) {

  }

   sendMove(): void {
    if (this.send) {
      const len = this.board.getMoveHistory().length;
      if (len == 0) {
        this.boardService.sendMove("start").subscribe(response => {
          this.send = false;
          this.board.move(response["move"]);
          this.send = true;
        })
      } else {
        const move = this.board.getMoveHistory()[this.board.getMoveHistory().length - 1];
        this.boardService.sendMove(move["move"]).subscribe(response => {
          this.send = false;
          this.board.move(response["move"]);
          this.send = true;
        })
      }
    }
  }

  ngOnInit(): void {

  }

  ngAfterViewInit(): void {
    this.sendMove();
  }

  undo(): void {
    this.board.undo();
  }
}
