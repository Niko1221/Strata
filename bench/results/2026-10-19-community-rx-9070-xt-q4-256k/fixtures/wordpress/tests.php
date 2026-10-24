<?php
const ARRAY_A='ARRAY_A';
$actions=[];$routes=[];$can=true;$filter_calls=0;
function add_action($hook,$cb){$GLOBALS['actions'][$hook][]=$cb;}
function register_rest_route($ns,$route,$args){$GLOBALS['routes'][]=[$ns,$route,$args];}
function current_user_can($cap){return $cap==='edit_posts'&&$GLOBALS['can'];}
function get_current_user_id(){return 7;}
function sanitize_text_field($s){return trim(strip_tags($s));}
function esc_html($s){return htmlspecialchars($s,ENT_QUOTES,'UTF-8');}
function apply_filters($hook,$value){if($hook==='devnotes_note'){$GLOBALS['filter_calls']++;$value['filtered']=true;}return $value;}
function __return_true(){return true;}
function is_wp_error($v){return $v instanceof WP_Error;}
class WP_Error {public function __construct(public $code,public $message='',public $data=[]){ } public function get_error_data(){return $this->data;}}
class WP_REST_Response {public function __construct(public $data,public $status=200){}public function get_data(){return $this->data;}public function get_status(){return $this->status;}}
class FakeDB {public $prefix='wp_',$insert_id=11,$fail=false,$last_error='sensitive-db-detail',$inserted=[],$prepared=false,$query='';
 function insert($table,$data,$format=null){$this->inserted=$data;return $this->fail?false:1;}
 function prepare($query,...$values){$this->prepared=true;return str_replace('%d',(string)$values[0],$query);}
 function get_results($query,$mode){$this->query=$query;return [['id'=>11,'label'=>'hello','owner_id'=>7]];}}
$wpdb=new FakeDB();require __DIR__.'/plugin.php';
$failures=0;$count=0;
function check($name,$fn){global $failures,$count;$count++;try{$fn();echo "PASS $name\n";}catch(Throwable $e){$failures++;echo "FAIL $name: ".$e->getMessage()."\n";}}
function expect($v,$message){if(!$v)throw new Exception($message);}
check('correct hook',function(){expect(isset($GLOBALS['actions']['rest_api_init']),'must register on rest_api_init');});
check('GET and POST permissions',function(){foreach($GLOBALS['actions']['rest_api_init']??[] as $cb)$cb();$methods=[];foreach($GLOBALS['routes'] as [$ns,$path,$args]){expect($ns==='devnotes/v1'&&$path==='/notes','route mismatch');foreach(isset($args['methods'])?[$args]:$args as $endpoint){if(!is_array($endpoint)||!isset($endpoint['methods']))continue;$methods[]=$endpoint['methods'];$GLOBALS['can']=false;expect(!call_user_func($endpoint['permission_callback'],[]),'permission must deny');$GLOBALS['can']=true;expect(call_user_func($endpoint['permission_callback'],[]),'permission must allow');}}expect(in_array('GET',$methods)&&in_array('POST',$methods),'both methods required');});
check('sanitize and owner isolation',function()use($wpdb){$rest=new DevNotesRest(new DevNotesRepository($wpdb));$r=$rest->create(['label'=>' <b>Hello</b> ','owner_id'=>99]);expect($r instanceof WP_REST_Response&&$r->get_status()===201,'201 response required');expect($wpdb->inserted['label']==='Hello','sanitize label');expect($wpdb->inserted['owner_id']===7,'owner is authenticated user');});
check('reject blank',function()use($wpdb){$r=(new DevNotesRest(new DevNotesRepository($wpdb)))->create(['label'=>'   ','owner_id'=>7]);expect(is_wp_error($r)&&($r->data['status']??0)===400,'blank must be 400');});
check('database failure',function()use($wpdb){$wpdb->fail=true;try{$r=(new DevNotesRest(new DevNotesRepository($wpdb)))->create(['label'=>'valid','owner_id'=>7]);expect(is_wp_error($r)&&($r->data['status']??0)===500,'DB failure must be 500');expect(!str_contains(json_encode($r),'sensitive-db-detail'),'must not expose DB error');}finally{$wpdb->fail=false;}});
check('prepared tenant-scoped listing and filter',function()use($wpdb){$r=(new DevNotesRest(new DevNotesRepository($wpdb)))->index([]);expect($wpdb->prepared&&preg_match('/owner_id\s*=\s*7/',$wpdb->query),'scope query using prepare');expect($GLOBALS['filter_calls']>0,'must apply note filter');});
check('escape admin output',function(){expect(!str_contains(devnotes_render(['label'=>'<script>alert(1)</script>']),'<script>'),'HTML must be escaped');});
echo json_encode(['tests'=>$count,'failed'=>$failures])."\n";exit($failures?1:0);
