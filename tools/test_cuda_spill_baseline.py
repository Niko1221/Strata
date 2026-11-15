"""A measured budget cannot silently follow source, flags or kernel growth."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from tools import cuda_spill_baseline as b, check_cuda_spills as check

class CudaSpillBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/"CMakeLists.txt").write_text("project(fixture)\n")
        (self.root/"src").mkdir()
        (self.root/"src"/"kernel.cu").write_bytes(b"kernel\r\n")
        self.key=self.root/"key.txt";self.key.write_text("a"*64)
        with mock.patch.object(b,"dependency_commit",return_value="b"*40):
            self.expected=b.bindings(self.root,self.key,self.root)
        self.file=self.root/"baseline.json"
        b.save(self.file,self.expected,{"lib|obj|sm_120|fn":(8,0)},[])

    def test_source_and_configuration_changes_are_rejected(self):
        self.assertEqual(b.load(self.file,self.expected)["lib|obj|sm_120|fn"],(8,0))
        for name in ["sourceDigest","compilerKey","ggmlCommit"]:
            changed={**self.expected,name:"changed"}
            with self.assertRaisesRegex(ValueError,"identity changed"):b.load(self.file,changed)
        before=b.source_digest(self.root)
        (self.root/"src"/"kernel.cu").write_bytes(b"kernel\n")
        self.assertEqual(b.source_digest(self.root),before,"line-ending conversion is not a kernel edit")
        (self.root/"src"/"kernel.cu").write_text("changed kernel\n")
        self.assertNotEqual(b.source_digest(self.root),before)

    def test_each_entry_is_stricter_than_family_allowance(self):
        func="mangled";names={func:"strata::kernels::{anonymous}::native_gu_kernel()"}
        key=b.entry_key("lib","obj","sm_120",func)
        rows=[("obj","sm_120",func,{"STACK":8,"LOCAL":0})]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check.check_measured("lib",rows,names,{key:(8,0)}),0)
            rows[0][3]["STACK"]=9
            self.assertEqual(check.check_measured("lib",rows,names,{key:(8,0)}),1)
            rows[0][3]["STACK"]=1
            self.assertEqual(check.check_measured("lib",rows,names,{}),1,"an unmeasured entry has no nonzero allowance")
            rows[0][3]["STACK"]=0
            self.assertEqual(check.check_measured("lib",rows,names,{}),0)

    def test_cli_never_uses_an_unbound_diagnostic_as_the_default_gate(self):
        library=self.root/"fixture.lib";library.write_bytes(b"fixture")
        name="strata::kernels::unmeasured_test_kernel()"
        rows=[("obj","sm_120","fn",{"STACK":16,"LOCAL":0})]
        known=self.root/"known.json"
        known.write_text(json.dumps({"sm_120":{"instances":[{
            "name":check.instance_key(name),"stack":16,"local":0,
            "reason":"Fixture diagnostic measurement"}]}}))
        before=known.read_bytes()
        with mock.patch.object(check,"find_tools",return_value=("cuobjdump","demangler")), \
             mock.patch.object(check,"resource_usage",return_value=rows), \
             mock.patch.object(check,"demangle",return_value={"fn":name}), \
             mock.patch.object(check,"BASELINE_FILE",str(known)):
            output=io.StringIO()
            with mock.patch("sys.argv",["check_cuda_spills.py",str(library)]), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(check.main(),1)
            self.assertIn("VIOLATION",output.getvalue())
            output=io.StringIO()
            with mock.patch("sys.argv",["check_cuda_spills.py",str(library),
                                      "--known-baseline",str(known)]), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(check.main(),0)
            self.assertIn("source-bound release qualification NOT_RUN",output.getvalue())
        self.assertEqual(known.read_bytes(),before)

    def test_cli_rejects_missing_bindings_and_mixed_diagnostic_mode_before_tools(self):
        cases=[
            ["lib","--baseline",str(self.file)],
            ["lib","--baseline",str(self.file),"--known-baseline","known.json"],
        ]
        for args in cases:
            with self.subTest(args=args), mock.patch("sys.argv",["check_cuda_spills.py",*args]), \
                 mock.patch.object(check,"find_tools") as tools, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                check.main()
            self.assertEqual(error.exception.code,2)
            tools.assert_not_called()

    def test_capture_refuses_to_overwrite_a_reviewed_file(self):
        with self.assertRaises(FileExistsError):b.save(self.file,self.expected,{},[])
        record=json.loads(self.file.read_text())
        record["entries"]["bad"]=[True,0]
        self.file.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,"resource amount"):b.load(self.file,self.expected)

    def test_vendored_dependency_identity_binds_all_content_and_not_parent_git(self):
        vendored=self.root/"vendor";(vendored/"ggml").mkdir(parents=True)
        (vendored/"UPSTREAM.md").write_text("Fixture pin\n")
        cmake=vendored/"ggml/CMakeLists.txt";cmake.write_bytes(b"project(ggml)\r\n")
        with mock.patch.object(b.subprocess,"check_output",side_effect=AssertionError("parent Git forbidden")):
            first=b.dependency_commit(vendored)
            self.assertTrue(first.startswith("vendored-sha256:"))
            cmake.write_bytes(b"project(ggml)\n")
            self.assertEqual(b.dependency_commit(vendored),first)
            extra=vendored/"ggml/kernel.cu";extra.write_text("new kernel\n")
            changed=b.dependency_commit(vendored)
            self.assertNotEqual(changed,first)
            extra.write_text("edited kernel\n")
            self.assertNotEqual(b.dependency_commit(vendored),changed)

if __name__=="__main__":unittest.main()
