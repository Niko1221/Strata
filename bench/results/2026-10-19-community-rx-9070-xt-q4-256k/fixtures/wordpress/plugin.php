<?php
require_once __DIR__.'/repository.php';
require_once __DIR__.'/rest.php';
add_action('init', function() { (new DevNotesRest(new DevNotesRepository($GLOBALS['wpdb'])))->register(); });
function devnotes_render($note) { return '<li>'.$note['label'].'</li>'; }
