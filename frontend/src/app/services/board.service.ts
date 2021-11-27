import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import {Observable} from 'rxjs';
const baseUrl='http://localhost:8080/api'
@Injectable({
  providedIn: 'root'
})
export class BoardService {

  constructor(private http: HttpClient) { }
    headers = new HttpHeaders({
    'Content-Type': 'application/json'
  });

  getNextMove(fen:String) : Observable<any> {
    const body = JSON.stringify({
      'fen':fen,
    });
    return this.http.post(baseUrl + '/next-move',body,{headers:this.headers});
  }

}
