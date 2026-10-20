<?php
class DevNotesRest {
 public function __construct(private $repo) {}
 public function register() { register_rest_route('devnotes/v1','/notes',['methods'=>'POST','callback'=>[$this,'create'],'permission_callback'=>'__return_true']); }
 public function create($request) { return $this->repo->create($request['label'],$request['owner_id']); }
 public function index($request) { return $this->repo->list_for(get_current_user_id()); }
}
