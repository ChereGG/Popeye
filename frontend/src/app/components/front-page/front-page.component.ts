import { Component, OnInit } from '@angular/core';
import {Router} from "@angular/router";

@Component({
  selector: 'app-front-page',
  templateUrl: './front-page.component.html',
  styleUrls: ['./front-page.component.css']
})
export class FrontPageComponent implements OnInit {

  constructor(private router:Router) { }

  ngOnInit(): void {
  }

  supervised() {
    this.router.navigateByUrl('/game', { state: {method:'supervised' } });
  }

  reinforcement() {
    this.router.navigateByUrl('/game', { state: {method:'reinforcement' } });

  }
}
