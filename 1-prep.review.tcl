
set candidate_files [glob 1-prep/*/candidates.tcl]
set num_candidates [llength $candidate_files]
set candidate_idx -1

proc show_next_candidate {} {
    global candidate_idx num_candidates
    incr candidate_idx
    if {$candidate_idx > $num_candidates} {
	set candidate_idx -1
    }
    show_candidate $candidate_idx
}

proc show_candidate {candidate_idx} {
    global candidate_files
    
    set f [lindex $candidate_files $candidate_idx]
    set d [file dirname "$f"]

    foreach ID [molinfo list] {
	mol delete $ID
    }
    
    set orig_dir [pwd]
    cd $d
    source candidates.tcl
    cd $orig_dir
}

proc show_previous_candidate {} {
    global candidate_idx num_candidates
    incr candidate_idx -1
    if {$candidate_idx < 0} {
	set candidate_idx [expr $num_candidates-1]
    }
    show_candidate $candidate_idx
}

user add key Control-n {
    show_next_candidate
}

user add key Control-p {
    show_previous_candidate
}

after idle { show_next_candidate }

set review_ch [open 1.prep.review.tsv a]

proc review {status msg} {
    global candidate_files candidate_idx review_ch
    regexp {.*/(.*)/candidates.tcl} [lindex $candidate_files $candidate_idx] --> pre
    puts $review_ch [join "$pre [string toupper $status] $msg" "\t"]
    show_next_candidate
}
