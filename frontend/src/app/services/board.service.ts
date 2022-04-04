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

  sendMove(move:String) : Observable<any> {
    const body = JSON.stringify({
      "move":move,
    });
    return this.http.post(baseUrl + '/send-move',body,{headers:this.headers});
  }

}
