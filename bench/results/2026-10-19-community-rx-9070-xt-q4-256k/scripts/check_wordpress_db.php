<?php
// Independent check outside the model's editable fixture.
const ARRAY_A='ARRAY_A';
function get_current_user_id(){return 7;}
function apply_filters($hook,$value){return $value;}
class WP_Error {public function __construct(public $code,public $message='',public $data=[]) {}}
class WP_REST_Response {public function __construct(public $data,public $status=200) {}}
class FailingReadDB {
 public $prefix='wp_';public $last_error='synthetic-database-failure';
 public function __construct(private bool $connectionUnavailable) {}
 function prepare($query,...$values){return str_replace('%d',(string)$values[0],$query);}
 // WordPress documents [] for SQL errors and null for an unavailable connection.
 function get_results($query,$mode){return $this->connectionUnavailable?null:[];}
}
$fixture=$argv[1]??'wordpress';
require $fixture.'/repository.php';
require $fixture.'/rest.php';
$failures=0;
foreach([false,true] as $connectionUnavailable) {
 $case=$connectionUnavailable?'connection unavailable':'SQL error';
 try {
  $result=(new DevNotesRest(new DevNotesRepository(new FailingReadDB($connectionUnavailable))))->index([]);
  if(!($result instanceof WP_Error)||($result->data['status']??0)!==500) {
   echo "FAIL $case: did not produce WP_Error 500\n";$failures++;
  } else {echo "PASS $case: WP_Error 500\n";}
 } catch(Throwable $e) {
  echo "FAIL $case: throws ".get_class($e)." instead of WP_Error 500\n";$failures++;
 }
}
exit($failures?1:0);
