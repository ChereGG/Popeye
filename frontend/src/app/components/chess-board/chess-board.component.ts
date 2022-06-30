import {Component, OnInit, ViewChild} from '@angular/core';
import {NgxChessBoardComponent, NgxChessBoardView} from 'ngx-chess-board';
import {BoardService} from "../../services/board.service";
import {HttpClient} from "@angular/common/http";
import {Location} from '@angular/common';
import {Router} from "@angular/router";

@Component({
  selector: 'app-chess-board',
  templateUrl: './chess-board.component.html',
  styleUrls: ['./chess-board.component.css']
})
export class ChessBoardComponent implements OnInit {
  @ViewChild(NgxChessBoardComponent) board: NgxChessBoardView;

  private send: boolean = true;
  public method: string;


  public constructor(private boardService: BoardService, private location: Location,private router:Router) {

  }

  sendMove(): void {
    if (this.method === 'reinforcement') {
      if (this.send) {
        const len = this.board.getMoveHistory().length;
        if (len == 0) {
          this.boardService.sendMoveReinforcement("start").subscribe(response => {
            this.send = false;
            this.board.move(response["move"]);
            this.send = true;
          })
        } else {
          const move = this.board.getMoveHistory()[this.board.getMoveHistory().length - 1];
          this.boardService.sendMoveReinforcement(move["move"]).subscribe(response => {
            this.send = false;
            this.board.move(response["move"]);
            this.send = true;
          })
        }
      }
    } else {
      if (this.send) {
        this.boardService.sendMoveSupervised(this.board.getFEN()).subscribe(response => {
          this.send = false;
          this.board.move(response["move"]);
          this.send = true;
        });
      }
    }
  }

  ngOnInit(): void {

    this.method = this.location.getState()['method'];
  }

  ngAfterViewInit(): void {
    this.sendMove();
  }

  undo(): void {
    this.board.undo();
    if (this.method === 'reinforcement') {
      this.boardService.sendUndoReinforcement(this.board.getMoveHistory()).subscribe(response => {
        this.send = false;
        this.board.move(response["move"]);
        this.send = true;
      });
    }
  }

  back() {
    this.router.navigate(['/'])
  }
}
