<?php
class DevNotesRepository {
 public function __construct(private $db) {}
 public function create($label,$owner) { $this->db->insert($this->db->prefix.'devnotes',['label'=>$label,'owner_id'=>$owner]); return ['id'=>$this->db->insert_id,'label'=>$label,'owner_id'=>$owner]; }
 public function list_for($owner) { return $this->db->get_results('SELECT * FROM '.$this->db->prefix.'devnotes', ARRAY_A); }
}
