from modules.ui.CloudTabController import CloudTabController
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType
from modules.util.ui import pyside6_components as ui

from PySide6.QtWidgets import QGroupBox, QWidget

# Label, control, config field, tooltip, and (for choices) displayed values.
# The order here is the order seen in each Cloud section.
CLOUD_SECTIONS = (
    ("Connection", (
        ("Enabled", "switch", "cloud.enabled", "Enable cloud training"),
        ("Type", "choice", "cloud.type", "Choose LINUX to connect to a linux machine via SSH. Choose RUNPOD for additional functionality such as automatically creating and deleting pods.", (("RUNPOD", CloudType.RUNPOD), ("LINUX", CloudType.LINUX))),
        ("File sync method", "choice", "cloud.file_sync", "Choose NATIVE_SCP to use scp.exe to transfer files. FABRIC_SFTP uses the Paramiko/Fabric SFTP implementation for file transfers instead.", (("NATIVE_SCP", CloudFileSync.NATIVE_SCP), ("FABRIC_SFTP", CloudFileSync.FABRIC_SFTP))),
        ("API key", "entry", "secrets.cloud.api_key", "Cloud service API key for RUNPOD. Leave empty for LINUX. This value is stored separately, not saved to your configuration file."),
        ("Hostname", "entry", "secrets.cloud.host", "SSH server hostname or IP. Leave empty if you have a Cloud ID or want to automatically create a new cloud."),
        ("Port", "entry", "secrets.cloud.port", "SSH server port. Leave empty if you have a Cloud ID or want to automatically create a new cloud."),
        ("User", "entry", "secrets.cloud.user", 'SSH username. Use "root" for RUNPOD. Your SSH client must be set up to connect to the cloud using a public key, without a password. For RUNPOD, create an ed25519 key locally, and copy the contents of the public keyfile to your "SSH Public Keys" on the RunPod website.'),
        ("SSH keyfile path", "path", "secrets.cloud.key_file", "Absolute path to the private key file used for SSH connections. Leave empty to rely on your system SSH configuration."),
        ("SSH password", "entry", "secrets.cloud.password", "SSH password for password-based authentication. If you try to use native SCP requires sshpass to be installed. Leave empty to use key-based authentication."),
        ("Cloud id", "entry", "secrets.cloud.id", "RUNPOD Cloud ID. The cloud service must have a public IP and SSH service. Leave empty if you want to automatically create a new RUNPOD cloud, or if you're connecting to another cloud provider via SSH Hostname and Port."),
    )),
    ("Remote installation", (
        ("Remote Directory", "entry", "cloud.remote_dir", "The directory on the cloud where files will be uploaded and downloaded."),
        ("OneTrainer Directory", "entry", "cloud.onetrainer_dir", "The directory for OneTrainer on the cloud."),
        ("Huggingface cache Directory", "entry", "cloud.huggingface_cache_dir", "Huggingface models are downloaded to this remote directory."),
        ("Install OneTrainer", "switch", "cloud.install_onetrainer", "Automatically install OneTrainer from GitHub if the directory doesn't already exist."),
        ("Install command", "entry", "cloud.install_cmd", "The command for installing OneTrainer. Leave the default, unless you want to use a development branch of OneTrainer."),
        ("Update OneTrainer", "switch", "cloud.update_onetrainer", "Update OneTrainer if it already exists on the cloud."),
    )),
    ("Create a cloud instance", (
        ("Create cloud via API", "create", "cloud.create", "Automatically creates a new cloud instance if both Host:Port and Cloud ID are empty. Currently supported for RUNPOD."),
        ("Cloud name", "entry", "cloud.name", "The name of the new cloud instance."),
        ("Type", "choice", "cloud.sub_type", "Select the RunPod cloud type. See RunPod's website for details.", (("", ""), ("Community", "COMMUNITY"), ("Secure", "SECURE"))),
        ("GPU", "gpu", "cloud.gpu_type", "Select the GPU type. Enter an API key before pressing the button."),
        ("Volume size", "entry", "cloud.volume_size", "Set the storage volume size in GB. This volume persists only until the cloud is deleted - not a RunPod network volume"),
        ("Min download", "entry", "cloud.min_download", "Set the minimum download speed of the cloud in Mbps."),
    )),
    ("Training connection", (
        ("Tensorboard TCP tunnel", "switch", "cloud.tensorboard_tunnel", "Instead of starting tensorboard locally, make a TCP tunnel to a tensorboard on the cloud"),
        ("Detach remote trainer", "switch", "cloud.detach_trainer", "Allows the trainer to keep running even if your connection to the cloud is lost."),
        ("Reattach id", "reattach", "cloud.run_id", "An id identifying the remotely running trainer. In case you have lost connection or closed OneTrainer, it will try to reattach to this id instead of starting a new remote trainer."),
    )),
    ("Download and cleanup", (
        ("Download samples", "switch", "cloud.download_samples", "Download samples from the remote workspace directory to your local machine."),
        ("Download output model", "switch", "cloud.download_output_model", "Download the final model after training. You can disable this if you plan to use an automatically saved checkpoint instead."),
        ("Download saved checkpoints", "switch", "cloud.download_saves", "Download the automatically saved training checkpoints from the remote workspace directory to your local machine."),
        ("Download backups", "switch", "cloud.download_backups", "Download backups from the remote workspace directory to your local machine. It's usually not necessary to download them, because as long as the backups are still available on the cloud, the training can be restarted using one of the cloud's backups."),
        ("Download tensorboard logs", "switch", "cloud.download_tensorboard", 'Download TensorBoard event logs from the remote workspace directory to your local machine. They can then be viewed locally in TensorBoard. It is recommended to disable "Sample to TensorBoard" to reduce the event log size.'),
        ("Delete remote workspace", "switch", "cloud.delete_workspace", "Delete the workspace directory on the cloud after training has finished successfully and data has been downloaded."),
    )),
    ("Instance lifecycle", (
        ("Action on finish", "choice", "cloud.on_finish", "What to do when training finishes and the data has been fully downloaded: Stop or delete the cloud, or do nothing.", (("None", CloudAction.NONE), ("Stop", CloudAction.STOP), ("Delete", CloudAction.DELETE))),
        ("Action on error", "choice", "cloud.on_error", "What to do if training stops due to an error: Stop or delete the cloud, or do nothing. Data may be lost.", (("None", CloudAction.NONE), ("Stop", CloudAction.STOP), ("Delete", CloudAction.DELETE))),
        ("Action on detached finish", "choice", "cloud.on_detached_finish", "What to do when training finishes, but the client has been detached and cannot download data. Data may be lost.", (("None", CloudAction.NONE), ("Stop", CloudAction.STOP), ("Delete", CloudAction.DELETE))),
        ("Action on detached error", "choice", "cloud.on_detached_error", "What to if training stops due to an error, but the client has been detached and cannot download data. Data may be lost.", (("None", CloudAction.NONE), ("Stop", CloudAction.STOP), ("Delete", CloudAction.DELETE))),
    )),
)


class PySide6CloudTabView(QWidget):
    @property
    def reattach(self):
        return self.controller.reattach

    def __init__(self, master, controller: CloudTabController, ui_state):
        super().__init__(master)
        self.controller = controller
        self.ui_state = ui_state
        scroll, self.frame = ui.scrollable_frame(self)
        ui._layout(self).addWidget(scroll, 0, 0)
        layout = ui._layout(self.frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(14)

        self._section_fields = {}
        self._section_groups = [self._build_section(title, fields) for title, fields in CLOUD_SECTIONS]
        self._section_columns = 0
        self._fields_stacked = None
        self._arrange_sections()

    def _build_section(self, title, fields):
        group = QGroupBox(title, self.frame)
        form = ui._layout(group)
        form.setContentsMargins(16, 18, 16, 16)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        form.setColumnStretch(1, 1)
        rows = []
        for row, (label, kind, name, tooltip, *choices) in enumerate(fields):
            caption = ui.label(group, row, 0, label, tooltip=tooltip)
            if kind == "switch":
                control = ui.switch(group, row, 1, self.ui_state, name)
            elif kind == "entry":
                control = ui.entry(group, row, 1, self.ui_state, name)
            elif kind == "path":
                control = ui.path_entry(group, row, 1, self.ui_state, name, mode="file")
            elif kind == "choice":
                control = ui.options_kv(group, row, 1, choices[0], self.ui_state, name)
            elif kind == "gpu":
                control, controls = ui.options_adv(
                    group, row, 1, [""], self.ui_state, name,
                    adv_command=self._on_set_gpu_types,
                )
                self.gpu_types_menu = controls["component"]
            elif kind == "reattach":
                control = ui.inline_frame(group, row, 1)
                ui._layout(control).setColumnStretch(0, 1)
                ui.entry(control, 0, 0, self.ui_state, name, width=60)
                ui.button(control, 0, 1, "Reattach now", self.controller.do_reattach)
            elif kind == "create":
                control = ui.inline_frame(group, row, 1)
                ui._layout(control).setColumnStretch(1, 1)
                ui.switch(control, 0, 0, self.ui_state, name)
                ui.button(control, 0, 1, "Create cloud via website", self.controller.open_create_cloud_url)
            rows.append((caption, control))
        self._section_fields[group] = rows
        return group

    def _arrange_sections(self):
        columns = 2 if self.width() >= 1040 else 1
        # A two-column page can be wide while each form is still too narrow
        # for its label and control on the same line.
        column_width = (self.width() - 24 - (14 if columns == 2 else 0)) / columns
        self._arrange_field_rows(column_width < 680)
        if columns == self._section_columns:
            return
        layout = ui._layout(self.frame)
        for group in self._section_groups:
            layout.removeWidget(group)
        for row in range(layout.rowCount() + 1):
            layout.setRowStretch(row, 0)
        for index, group in enumerate(self._section_groups):
            layout.addWidget(group, index // columns, index % columns)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1 if columns == 2 else 0)
        layout.setRowStretch((len(self._section_groups) + columns - 1) // columns, 1)
        self._section_columns = columns

    def _arrange_field_rows(self, stacked):
        if stacked == self._fields_stacked:
            return
        for group, rows in self._section_fields.items():
            layout = ui._layout(group)
            for caption, control in rows:
                layout.removeWidget(caption)
                layout.removeWidget(control)
            for index, (caption, control) in enumerate(rows):
                if stacked:
                    layout.addWidget(caption, index * 2, 0)
                    layout.addWidget(control, index * 2 + 1, 0)
                else:
                    layout.addWidget(caption, index, 0)
                    layout.addWidget(control, index, 1)
            layout.setColumnStretch(0, 1 if stacked else 0)
            layout.setColumnStretch(1, 0 if stacked else 1)
            layout.invalidate()
            group.updateGeometry()
        ui._layout(self.frame).invalidate()
        self.frame.updateGeometry()
        self._fields_stacked = stacked

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_section_groups"):
            self._arrange_sections()

    def _on_set_gpu_types(self):
        self.gpu_types_menu.clear()
        self.gpu_types_menu.addItems(self.controller.get_gpu_types())
