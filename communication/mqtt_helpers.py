import asyncio
import paho.mqtt.client as mqtt

class AIOSocket:
    def __init__(self, loop, client, on_disconnect):
        """
        Initialize an instance of AIOSocket.

        This class manages the asynchronous socket operations for an MQTT client using asyncio and paho-mqtt library.

        Parameters:
        - loop (asyncio.BaseEventLoop): The event loop to run the asynchronous tasks.
        - client (mqtt.Client): The MQTT client instance to manage the socket operations.
        - on_disconnect (callable): A callback function to be called when the socket is disconnected.
            The callback function should accept three parameters: client, userdata, and rc (return code).

        Attributes:
        - loop (asyncio.BaseEventLoop): The event loop used for asynchronous tasks.
        - client (mqtt.Client): The MQTT client instance.
        - client.on_socket_open (callable): A callback function to be called when the socket is opened.
        - client.on_socket_close (callable): A callback function to be called when the socket is closed.
        - client.on_socket_register_write (callable): A callback function to be called when the socket is ready for writing.
        - client.on_socket_unregister_write (callable): A callback function to be called when the socket is no longer ready for writing.
        - on_disconnect (callable): The callback function to be called when the socket is disconnected.
        - is_connected (bool): A flag indicating whether the socket is currently connected.
        - last_rc (int): The last return code received from the MQTT client.

        Returns:
        None
        """
        self.loop = loop
        self.client = client
        self.client.on_socket_open = self.on_socket_open
        self.client.on_socket_close = self.on_socket_close
        self.client.on_socket_register_write = self.on_socket_register_write
        self.client.on_socket_unregister_write = self.on_socket_unregister_write
        self.on_disconnect = on_disconnect
        self.is_connected = False
        self.last_rc = 0

    def on_socket_open(self, client, userdata, sock):
        def cb():
            client.loop_read()

        self.loop.add_reader(sock, cb)
        self.misc = self.loop.create_task(self.misc_loop())
        self.is_connected = True

    def on_socket_close(self, client, userdata, sock):
        self.loop.remove_reader(sock)
        self.misc.cancel()
        self.is_connected = False
        if self.on_disconnect:
            # Pass the required arguments to the callback
            self.on_disconnect(client, userdata, self.last_rc)

    def on_socket_register_write(self, client, userdata, sock):
        def cb():
            client.loop_write()

        self.loop.add_writer(sock, cb)

    def on_socket_unregister_write(self, client, userdata, sock):
        self.loop.remove_writer(sock)

    async def misc_loop(self):
        while self.client.loop_misc() == mqtt.MQTT_ERR_SUCCESS:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break