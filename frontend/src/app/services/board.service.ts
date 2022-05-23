import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import {Observable} from 'rxjs';
import {HistoryMove} from "ngx-chess-board";
const baseUrl='http://localhost:8080/api'
@Injectable({
  providedIn: 'root'
})
export class BoardService {

  constructor(private http: HttpClient) { }
    headers = new HttpHeaders({
    'Content-Type': 'application/json'
  });

  sendMoveReinforcement(move:String) : Observable<any> {
    const body = JSON.stringify({
      "move":move,
    });
    return this.http.post(baseUrl + '/send-move-reinforcement',body,{headers:this.headers});
  }

  sendMoveSupervised(fen:String) : Observable<any> {
    const body = JSON.stringify({
      "fen":fen,
    });
    return this.http.post(baseUrl + '/send-move-supervised',body,{headers:this.headers});
  }

  sendUndoReinforcement(move_history: HistoryMove[]): Observable<any> {
    let moves=[];
    for (let move in move_history){
      moves.push(move_history[move]['move']);
    }
    const body = JSON.stringify({
      "move_history":moves,
    });
    return this.http.post(baseUrl + '/undo-reinforcement',body,{headers:this.headers})
  }
}
